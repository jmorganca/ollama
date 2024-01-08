package llm

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/jmorganca/ollama/format"
)

type containerGGUF struct {
	bo binary.ByteOrder

	Version uint32

	V1 struct {
		NumTensor uint32
		NumKV     uint32
	}

	V2 struct {
		NumTensor uint64
		NumKV     uint64
	}
}

func (c *containerGGUF) Name() string {
	return "gguf"
}

func (c *containerGGUF) Decode(rso *readSeekOffset) (model, error) {
	binary.Read(rso, c.bo, &c.Version)

	switch c.Version {
	case 1:
		binary.Read(rso, c.bo, &c.V1)
	default:
		binary.Read(rso, c.bo, &c.V2)
	}

	model := newGGUFModel(c)
	if err := model.Decode(rso); err != nil {
		return nil, err
	}

	return model, nil
}

const (
	ggufTypeUint8 uint32 = iota
	ggufTypeInt8
	ggufTypeUint16
	ggufTypeInt16
	ggufTypeUint32
	ggufTypeInt32
	ggufTypeFloat32
	ggufTypeBool
	ggufTypeString
	ggufTypeArray
	ggufTypeUint64
	ggufTypeInt64
	ggufTypeFloat64
)

type kv map[string]any

type tensor struct {
	name   string
	kind   uint32
	offset uint64
	size   uint64

	// shape is the number of elements in each dimension
	shape [4]uint64
}

type ggufModel struct {
	*containerGGUF

	kv
	tensors []tensor

	parameters uint64
}

func newGGUFModel(container *containerGGUF) *ggufModel {
	return &ggufModel{
		containerGGUF: container,
		kv:            make(kv),
	}
}

func (llm *ggufModel) NumTensor() uint64 {
	if llm.Version == 1 {
		return uint64(llm.V1.NumTensor)
	}

	return llm.V2.NumTensor
}

func (llm *ggufModel) NumKV() uint64 {
	if llm.Version == 1 {
		return uint64(llm.V1.NumKV)
	}

	return llm.V2.NumKV
}

func (llm *ggufModel) ModelFamily() string {
	if t, ok := llm.kv["general.architecture"].(string); ok {
		return t
	}

	return "unknown"
}

func (llm *ggufModel) ModelType() string {
	if llm.parameters > 0 {
		return format.HumanNumber(llm.parameters)
	}

	return "unknown"
}

func (llm *ggufModel) FileType() string {
	if t, ok := llm.kv["general.file_type"].(uint32); ok {
		return fileType(t)
	}

	return "unknown"
}

func (llm *ggufModel) Decode(rso *readSeekOffset) error {
	// decode key-values
	for i := 0; uint64(i) < llm.NumKV(); i++ {
		k, err := llm.readString(rso)
		if err != nil {
			return err
		}

		vtype := llm.readU32(rso)

		var v any
		switch vtype {
		case ggufTypeUint8:
			v = llm.readU8(rso)
		case ggufTypeInt8:
			v = llm.readI8(rso)
		case ggufTypeUint16:
			v = llm.readU16(rso)
		case ggufTypeInt16:
			v = llm.readI16(rso)
		case ggufTypeUint32:
			v = llm.readU32(rso)
		case ggufTypeInt32:
			v = llm.readI32(rso)
		case ggufTypeUint64:
			v = llm.readU64(rso)
		case ggufTypeInt64:
			v = llm.readI64(rso)
		case ggufTypeFloat32:
			v = llm.readF32(rso)
		case ggufTypeFloat64:
			v = llm.readF64(rso)
		case ggufTypeBool:
			v = llm.readBool(rso)
		case ggufTypeString:
			s, err := llm.readString(rso)
			if err != nil {
				return err
			}

			v = s
		case ggufTypeArray:
			a, err := llm.readArray(rso)
			if err != nil {
				return err
			}

			v = a
		default:
			return fmt.Errorf("invalid type: %d", vtype)
		}

		llm.kv[k] = v
	}

	// decode tensors
	for i := 0; uint64(i) < llm.NumTensor(); i++ {
		name, err := llm.readString(rso)
		if err != nil {
			return err
		}

		// dims is the number of dimensions in the tensor
		dims := llm.readU32(rso)

		shape := [4]uint64{1, 1, 1, 1}
		for i := 0; uint32(i) < dims; i++ {
			shape[i] = llm.readU64(rso)
		}

		kind := llm.readU32(rso)
		offset := llm.readU64(rso)

		var blockSize uint64
		switch {
		case kind < 2:
			blockSize = 1
		case kind < 10:
			blockSize = 32
		default:
			blockSize = 256
		}

		var typeSize uint64
		switch kind {
		case 0: // FP32
			typeSize = 4
		case 1: // FP16
			typeSize = 2
		case 2: // Q4_0
			typeSize = 2 + blockSize/2
		case 3: // Q4_1
			typeSize = 2 + 2 + blockSize/2
		case 6: // Q5_0
			typeSize = 2 + 4 + blockSize/2
		case 7: // Q5_1
			typeSize = 2 + 2 + 4 + blockSize/2
		case 8: // Q8_0
			typeSize = 2 + blockSize
		case 9: // Q8_1
			typeSize = 4 + 4 + blockSize
		case 10: // Q2_K
			typeSize = blockSize/16 + blockSize/4 + 2 + 2
		case 11: // Q3_K
			typeSize = blockSize/8 + blockSize/4 + 12 + 2
		case 12: // Q4_K
			typeSize = 2 + 2 + 12 + blockSize/2
		case 13: // Q5_K
			typeSize = 2 + 2 + 12 + blockSize/8 + blockSize/2
		case 14: // Q6_K
			typeSize = blockSize/2 + blockSize/4 + blockSize/16 + 2
		}

		parameters := shape[0] * shape[1] * shape[2] * shape[3]
		size := parameters * typeSize / blockSize

		llm.tensors = append(llm.tensors, tensor{
			name:   name,
			kind:   kind,
			offset: offset,
			size:   size,
			shape:  shape,
		})

		llm.parameters += parameters
	}

	alignment, ok := llm.kv["general.alignment"].(uint32)
	if !ok {
		alignment = 32
	}

	rso.Seek(int64(alignment)-rso.offset%int64(alignment), io.SeekCurrent)
	for _, tensor := range llm.tensors {
		padded := (int64(tensor.size) + int64(alignment) - 1) & ^(int64(alignment) - 1)
		rso.Seek(padded, io.SeekCurrent)
	}

	return nil
}

func (llm *ggufModel) NumLayers() uint32 {
	value, exists := llm.kv[fmt.Sprintf("%s.block_count", llm.ModelFamily())]
	if !exists {
		return 0
	}

	return value.(uint32)
}

func (llm *ggufModel) NumHead() uint32 {
	value, exists := llm.kv[fmt.Sprintf("%s.attention.head_count", llm.ModelFamily())]
	if !exists {
		return 0
	}

	return value.(uint32)
}

func (llm *ggufModel) NumEmbed() uint32 {
	value, exists := llm.kv[fmt.Sprintf("%s.embedding_length", llm.ModelFamily())]
	if !exists {
		return 0
	}

	return value.(uint32)
}

func (llm *ggufModel) NumHeadKv() uint32 {
	value, exists := llm.kv[fmt.Sprintf("%s.attention.head_count_kv", llm.ModelFamily())]
	if !exists {
		return 0
	}

	return value.(uint32)
}

func (llm *ggufModel) NumGQA() uint32 {
	numHeadKv := llm.NumHeadKv()
	if numHeadKv == 0 {
		return 0
	}

	return llm.NumHead() / numHeadKv
}

func (llm ggufModel) readU8(r io.Reader) uint8 {
	var u8 uint8
	binary.Read(r, llm.bo, &u8)
	return u8
}

func (llm ggufModel) readI8(r io.Reader) int8 {
	var i8 int8
	binary.Read(r, llm.bo, &i8)
	return i8
}

func (llm ggufModel) readU16(r io.Reader) uint16 {
	var u16 uint16
	binary.Read(r, llm.bo, &u16)
	return u16
}

func (llm ggufModel) readI16(r io.Reader) int16 {
	var i16 int16
	binary.Read(r, llm.bo, &i16)
	return i16
}

func (llm ggufModel) readU32(r io.Reader) uint32 {
	var u32 uint32
	binary.Read(r, llm.bo, &u32)
	return u32
}

func (llm ggufModel) readI32(r io.Reader) int32 {
	var i32 int32
	binary.Read(r, llm.bo, &i32)
	return i32
}

func (llm ggufModel) readU64(r io.Reader) uint64 {
	var u64 uint64
	binary.Read(r, llm.bo, &u64)
	return u64
}

func (llm ggufModel) readI64(r io.Reader) int64 {
	var i64 int64
	binary.Read(r, llm.bo, &i64)
	return i64
}

func (llm ggufModel) readF32(r io.Reader) float32 {
	var f32 float32
	binary.Read(r, llm.bo, &f32)
	return f32
}

func (llm ggufModel) readF64(r io.Reader) float64 {
	var f64 float64
	binary.Read(r, llm.bo, &f64)
	return f64
}

func (llm ggufModel) readBool(r io.Reader) bool {
	var b bool
	binary.Read(r, llm.bo, &b)
	return b
}

func (llm ggufModel) readStringV1(r io.Reader) (string, error) {
	var nameLength uint32
	binary.Read(r, llm.bo, &nameLength)

	var b bytes.Buffer
	if _, err := io.CopyN(&b, r, int64(nameLength)); err != nil {
		return "", err
	}

	// gguf v1 strings are null-terminated
	b.Truncate(b.Len() - 1)

	return b.String(), nil
}

func (llm ggufModel) readString(r io.Reader) (string, error) {
	if llm.Version == 1 {
		return llm.readStringV1(r)
	}

	var nameLength uint64
	binary.Read(r, llm.bo, &nameLength)

	var b bytes.Buffer
	if _, err := io.CopyN(&b, r, int64(nameLength)); err != nil {
		return "", err
	}

	return b.String(), nil
}

func (llm *ggufModel) readArrayV1(r io.Reader) (arr []any, err error) {
	atype := llm.readU32(r)
	n := llm.readU32(r)

	for i := 0; uint32(i) < n; i++ {
		switch atype {
		case ggufTypeUint8:
			arr = append(arr, llm.readU8(r))
		case ggufTypeInt8:
			arr = append(arr, llm.readI8(r))
		case ggufTypeUint16:
			arr = append(arr, llm.readU16(r))
		case ggufTypeInt16:
			arr = append(arr, llm.readI16(r))
		case ggufTypeUint32:
			arr = append(arr, llm.readU32(r))
		case ggufTypeInt32:
			arr = append(arr, llm.readI32(r))
		case ggufTypeFloat32:
			arr = append(arr, llm.readF32(r))
		case ggufTypeBool:
			arr = append(arr, llm.readBool(r))
		case ggufTypeString:
			s, err := llm.readStringV1(r)
			if err != nil {
				return nil, err
			}

			arr = append(arr, s)
		default:
			return nil, fmt.Errorf("invalid array type: %d", atype)
		}
	}

	return
}

func (llm *ggufModel) readArray(r io.Reader) (arr []any, err error) {
	if llm.Version == 1 {
		return llm.readArrayV1(r)
	}

	atype := llm.readU32(r)
	n := llm.readU64(r)

	for i := 0; uint64(i) < n; i++ {
		switch atype {
		case ggufTypeUint8:
			arr = append(arr, llm.readU8(r))
		case ggufTypeInt8:
			arr = append(arr, llm.readI8(r))
		case ggufTypeUint16:
			arr = append(arr, llm.readU16(r))
		case ggufTypeInt16:
			arr = append(arr, llm.readI16(r))
		case ggufTypeUint32:
			arr = append(arr, llm.readU32(r))
		case ggufTypeInt32:
			arr = append(arr, llm.readI32(r))
		case ggufTypeUint64:
			arr = append(arr, llm.readU64(r))
		case ggufTypeInt64:
			arr = append(arr, llm.readI64(r))
		case ggufTypeFloat32:
			arr = append(arr, llm.readF32(r))
		case ggufTypeFloat64:
			arr = append(arr, llm.readF64(r))
		case ggufTypeBool:
			arr = append(arr, llm.readBool(r))
		case ggufTypeString:
			s, err := llm.readString(r)
			if err != nil {
				return nil, err
			}

			arr = append(arr, s)
		default:
			return nil, fmt.Errorf("invalid array type: %d", atype)
		}
	}

	return
}
