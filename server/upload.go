package server

import (
	"context"
	"crypto/md5"
	"errors"
	"fmt"
	"hash"
	"io"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/format"
	"golang.org/x/sync/errgroup"
)

var blobUploadManager sync.Map

type blobUpload struct {
	*Layer

	Total     int64
	Completed atomic.Int64

	Parts []blobUploadPart

	nextURL chan *url.URL

	context.CancelFunc

	file *os.File

	done       bool
	err        error
	references atomic.Int32
}

const (
	numUploadParts          = 64
	minUploadPartSize int64 = 100 * format.MegaByte
	maxUploadPartSize int64 = 1000 * format.MegaByte
)

func (b *blobUpload) Prepare(ctx context.Context, requestURL *url.URL, opts *RegistryOptions) error {
	p, err := GetBlobsPath(b.Digest)
	if err != nil {
		return err
	}

	if b.From != "" {
		values := requestURL.Query()
		values.Add("mount", b.Digest)
		values.Add("from", ParseModelPath(b.From).GetNamespaceRepository())
		requestURL.RawQuery = values.Encode()
	}

	resp, err := makeRequestWithRetry(ctx, http.MethodPost, requestURL, nil, nil, opts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	location := resp.Header.Get("Docker-Upload-Location")
	if location == "" {
		location = resp.Header.Get("Location")
	}

	fi, err := os.Stat(p)
	if err != nil {
		return err
	}

	b.Total = fi.Size()

	// http.StatusCreated indicates a blob has been mounted
	// ref: https://distribution.github.io/distribution/spec/api/#cross-repository-blob-mount
	if resp.StatusCode == http.StatusCreated {
		b.Completed.Store(b.Total)
		b.done = true
		return nil
	}

	size := b.Total / numUploadParts
	switch {
	case size < minUploadPartSize:
		size = minUploadPartSize
	case size > maxUploadPartSize:
		size = maxUploadPartSize
	}

	var offset int64
	for offset < fi.Size() {
		if offset+size > fi.Size() {
			size = fi.Size() - offset
		}

		// set part.N to the current number of parts
		b.Parts = append(b.Parts, blobUploadPart{N: len(b.Parts), Offset: offset, Size: size})
		offset += size
	}

	log.Printf("uploading %s in %d %s part(s)", b.Digest[7:19], len(b.Parts), format.HumanBytes(b.Parts[0].Size))

	requestURL, err = url.Parse(location)
	if err != nil {
		return err
	}

	b.nextURL = make(chan *url.URL, 1)
	b.nextURL <- requestURL
	return nil
}

// Run uploads blob parts to the upstream. If the upstream supports redirection, parts will be uploaded
// in parallel as defined by Prepare. Otherwise, parts will be uploaded serially. Run sets b.err on error.
func (b *blobUpload) Run(ctx context.Context, opts *RegistryOptions) {
	defer blobUploadManager.Delete(b.Digest)
	ctx, b.CancelFunc = context.WithCancel(ctx)

	p, err := GetBlobsPath(b.Digest)
	if err != nil {
		b.err = err
		return
	}

	b.file, err = os.Open(p)
	if err != nil {
		b.err = err
		return
	}
	defer b.file.Close()

	g, inner := errgroup.WithContext(ctx)
	g.SetLimit(numUploadParts)
	for i := range b.Parts {
		part := &b.Parts[i]
		select {
		case <-inner.Done():
		case requestURL := <-b.nextURL:
			g.Go(func() error {
				var err error
				for try := 0; try < maxRetries; try++ {
					err = b.uploadPart(inner, http.MethodPatch, requestURL, part, opts)
					switch {
					case errors.Is(err, context.Canceled):
						return err
					case errors.Is(err, errMaxRetriesExceeded):
						return err
					case err != nil:
						sleep := time.Second * time.Duration(math.Pow(2, float64(try)))
						log.Printf("%s part %d attempt %d failed: %v, retrying in %s", b.Digest[7:19], part.N, try, err, sleep)
						time.Sleep(sleep)
						continue
					}

					return nil
				}

				return fmt.Errorf("%w: %w", errMaxRetriesExceeded, err)
			})
		}
	}

	if err := g.Wait(); err != nil {
		b.err = err
		return
	}

	requestURL := <-b.nextURL

	// calculate md5 checksum and add it to the commit request
	var sb strings.Builder
	for _, part := range b.Parts {
		sb.Write(part.Sum(nil))
	}

	md5sum := md5.Sum([]byte(sb.String()))

	values := requestURL.Query()
	values.Add("digest", b.Digest)
	values.Add("etag", fmt.Sprintf("%x-%d", md5sum, len(b.Parts)))
	requestURL.RawQuery = values.Encode()

	headers := make(http.Header)
	headers.Set("Content-Type", "application/octet-stream")
	headers.Set("Content-Length", "0")

	for try := 0; try < maxRetries; try++ {
		var resp *http.Response
		resp, err = makeRequestWithRetry(ctx, http.MethodPut, requestURL, headers, nil, opts)
		if errors.Is(err, context.Canceled) {
			break
		} else if err != nil {
			sleep := time.Second * time.Duration(math.Pow(2, float64(try)))
			log.Printf("%s complete upload attempt %d failed: %v, retrying in %s", b.Digest[7:19], try, err, sleep)
			time.Sleep(sleep)
			continue
		}
		defer resp.Body.Close()
		break
	}

	b.err = err
	b.done = true
}

func (b *blobUpload) uploadPart(ctx context.Context, method string, requestURL *url.URL, part *blobUploadPart, opts *RegistryOptions) error {
	headers := make(http.Header)
	headers.Set("Content-Type", "application/octet-stream")
	headers.Set("Content-Length", fmt.Sprintf("%d", part.Size))

	if method == http.MethodPatch {
		headers.Set("X-Redirect-Uploads", "1")
		headers.Set("Content-Range", fmt.Sprintf("%d-%d", part.Offset, part.Offset+part.Size-1))
	}

	sr := io.NewSectionReader(b.file, part.Offset, part.Size)

	md5sum := md5.New()
	w := &progressWriter{blobUpload: b}

	resp, err := makeRequest(ctx, method, requestURL, headers, io.TeeReader(sr, io.MultiWriter(w, md5sum)), opts)
	if err != nil {
		w.Rollback()
		return err
	}
	defer resp.Body.Close()

	location := resp.Header.Get("Docker-Upload-Location")
	if location == "" {
		location = resp.Header.Get("Location")
	}

	nextURL, err := url.Parse(location)
	if err != nil {
		w.Rollback()
		return err
	}

	switch {
	case resp.StatusCode == http.StatusTemporaryRedirect:
		w.Rollback()
		b.nextURL <- nextURL

		redirectURL, err := resp.Location()
		if err != nil {
			return err
		}

		// retry uploading to the redirect URL
		for try := 0; try < maxRetries; try++ {
			err = b.uploadPart(ctx, http.MethodPut, redirectURL, part, nil)
			switch {
			case errors.Is(err, context.Canceled):
				return err
			case errors.Is(err, errMaxRetriesExceeded):
				return err
			case err != nil:
				sleep := time.Second * time.Duration(math.Pow(2, float64(try)))
				log.Printf("%s part %d attempt %d failed: %v, retrying in %s", b.Digest[7:19], part.N, try, err, sleep)
				time.Sleep(sleep)
				continue
			}

			return nil
		}

		return fmt.Errorf("%w: %w", errMaxRetriesExceeded, err)

	case resp.StatusCode == http.StatusUnauthorized:
		w.Rollback()
		auth := resp.Header.Get("www-authenticate")
		authRedir := ParseAuthRedirectString(auth)
		token, err := getAuthToken(ctx, authRedir)
		if err != nil {
			return err
		}

		opts.Token = token
		fallthrough
	case resp.StatusCode >= http.StatusBadRequest:
		w.Rollback()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}

		return fmt.Errorf("http status %s: %s", resp.Status, body)
	}

	if method == http.MethodPatch {
		b.nextURL <- nextURL
	}

	part.Hash = md5sum
	return nil
}

func (b *blobUpload) acquire() {
	b.references.Add(1)
}

func (b *blobUpload) release() {
	if b.references.Add(-1) == 0 {
		b.CancelFunc()
	}
}

func (b *blobUpload) Wait(ctx context.Context, fn func(api.ProgressResponse)) error {
	b.acquire()
	defer b.release()

	ticker := time.NewTicker(60 * time.Millisecond)
	for {
		select {
		case <-ticker.C:
		case <-ctx.Done():
			return ctx.Err()
		}

		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("pushing %s", b.Digest[7:19]),
			Digest:    b.Digest,
			Total:     b.Total,
			Completed: b.Completed.Load(),
		})

		if b.done || b.err != nil {
			return b.err
		}
	}
}

type blobUploadPart struct {
	// N is the part number
	N      int
	Offset int64
	Size   int64
	hash.Hash
}

type progressWriter struct {
	written int64
	*blobUpload
}

func (p *progressWriter) Write(b []byte) (n int, err error) {
	n = len(b)
	p.written += int64(n)
	p.Completed.Add(int64(n))
	return n, nil
}

func (p *progressWriter) Rollback() {
	p.Completed.Add(-p.written)
	p.written = 0
}

func uploadBlob(ctx context.Context, mp ModelPath, layer *Layer, opts *RegistryOptions, fn func(api.ProgressResponse)) error {
	requestURL := mp.BaseURL()
	requestURL = requestURL.JoinPath("v2", mp.GetNamespaceRepository(), "blobs", layer.Digest)

	resp, err := makeRequestWithRetry(ctx, http.MethodHead, requestURL, nil, nil, opts)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		return err
	default:
		defer resp.Body.Close()
		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("pushing %s", layer.Digest[7:19]),
			Digest:    layer.Digest,
			Total:     layer.Size,
			Completed: layer.Size,
		})

		return nil
	}

	data, ok := blobUploadManager.LoadOrStore(layer.Digest, &blobUpload{Layer: layer})
	upload := data.(*blobUpload)
	if !ok {
		requestURL := mp.BaseURL()
		requestURL = requestURL.JoinPath("v2", mp.GetNamespaceRepository(), "blobs/uploads/")
		if err := upload.Prepare(ctx, requestURL, opts); err != nil {
			blobUploadManager.Delete(layer.Digest)
			return err
		}

		// nolint: contextcheck
		go upload.Run(context.Background(), opts)
	}

	return upload.Wait(ctx, fn)
}
