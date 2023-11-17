package editor

import (
	"errors"
)

var (
	ErrInterrupt = errors.New("Interrupt")
)

type InterruptError struct {
	Line []rune
}

func (*InterruptError) Error() string {
	return "Interrupted"
}
