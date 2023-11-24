package llm

import (
	"encoding/binary"
	"errors"
	"io"
)

type GGML struct {
	container
	model

	Size int64
}

const (
	fileTypeF32 uint32 = iota
	fileTypeF16
	fileTypeQ4_0
	fileTypeQ4_1
	fileTypeQ4_1_F16
	fileTypeQ8_0 uint32 = iota + 2
	fileTypeQ5_0
	fileTypeQ5_1
	fileTypeQ2_K
	fileTypeQ3_K_S
	fileTypeQ3_K_M
	fileTypeQ3_K_L
	fileTypeQ4_K_S
	fileTypeQ4_K_M
	fileTypeQ5_K_S
	fileTypeQ5_K_M
	fileTypeQ6_K
)

func fileType(fileType uint32) string {
	switch fileType {
	case fileTypeF32:
		return "F32"
	case fileTypeF16:
		return "F16"
	case fileTypeQ4_0:
		return "Q4_0"
	case fileTypeQ4_1:
		return "Q4_1"
	case fileTypeQ4_1_F16:
		return "Q4_1_F16"
	case fileTypeQ8_0:
		return "Q8_0"
	case fileTypeQ5_0:
		return "Q5_0"
	case fileTypeQ5_1:
		return "Q5_1"
	case fileTypeQ2_K:
		return "Q2_K"
	case fileTypeQ3_K_S:
		return "Q3_K_S"
	case fileTypeQ3_K_M:
		return "Q3_K_M"
	case fileTypeQ3_K_L:
		return "Q3_K_L"
	case fileTypeQ4_K_S:
		return "Q4_K_S"
	case fileTypeQ4_K_M:
		return "Q4_K_M"
	case fileTypeQ5_K_S:
		return "Q5_K_S"
	case fileTypeQ5_K_M:
		return "Q5_K_M"
	case fileTypeQ6_K:
		return "Q6_K"
	default:
		return "unknown"
	}
}

type model interface {
	ModelFamily() string
	ModelType() string
	FileType() string
	NumLayers() int64
}

type container interface {
	Name() string
	Decode(*readSeekOffset) (model, error)
}

type containerLORA struct {
	version uint32
}

func (c *containerLORA) Name() string {
	return "ggla"
}

func (c *containerLORA) Decode(ro *readSeekOffset) (model, error) {
	var version uint32
	binary.Read(ro, binary.LittleEndian, &version)

	switch version {
	case 1:
	default:
		return nil, errors.New("invalid version")
	}

	c.version = version

	// remaining file contents aren't decoded
	ro.Seek(0, io.SeekEnd)

	return nil, nil
}

const (
	// Magic constant for `ggml` files (unversioned).
	FILE_MAGIC_GGML = 0x67676d6c
	// Magic constant for `ggml` files (versioned, ggmf).
	FILE_MAGIC_GGMF = 0x67676d66
	// Magic constant for `ggml` files (versioned, ggjt).
	FILE_MAGIC_GGJT = 0x67676a74
	// Magic constant for `ggla` files (LoRA adapter).
	FILE_MAGIC_GGLA = 0x67676C61
	// Magic constant for `gguf` files (versioned, gguf)
	FILE_MAGIC_GGUF_LE = 0x46554747
	FILE_MAGIC_GGUF_BE = 0x47475546
)

var ErrUnsupportedFormat = errors.New("unsupported model format")

func DecodeGGML(r io.ReadSeeker) (*GGML, error) {
	ro := readSeekOffset{ReadSeeker: r}

	var magic uint32
	if err := binary.Read(&ro, binary.LittleEndian, &magic); err != nil {
		return nil, err
	}

	var c container
	switch magic {
	case FILE_MAGIC_GGML, FILE_MAGIC_GGMF, FILE_MAGIC_GGJT:
		return nil, ErrUnsupportedFormat
	case FILE_MAGIC_GGLA:
		c = &containerLORA{}
	case FILE_MAGIC_GGUF_LE:
		c = &containerGGUF{bo: binary.LittleEndian}
	case FILE_MAGIC_GGUF_BE:
		c = &containerGGUF{bo: binary.BigEndian}
	default:
		return nil, errors.New("invalid file magic")
	}

	model, err := c.Decode(&ro)
	if err != nil {
		return nil, err
	}

	// final model type
	return &GGML{
		container: c,
		model:     model,
		Size:      ro.offset,
	}, nil
}

type readSeekOffset struct {
	io.ReadSeeker
	offset int64
}

func (rso *readSeekOffset) Seek(offset int64, whence int) (int64, error) {
	offset, err := rso.ReadSeeker.Seek(offset, whence)
	if err != nil {
		return 0, err
	}

	rso.offset = offset
	return offset, nil
}

func (rso *readSeekOffset) Read(p []byte) (int, error) {
	n, err := rso.ReadSeeker.Read(p)
	rso.offset += int64(n)
	return n, err
}
