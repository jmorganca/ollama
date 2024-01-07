//go:build !windows

package readline

import (
	"syscall"
)

func handleCharCtrlZ(fd int, termios *Termios) (string, error) {
	if err := UnsetRawMode(fd, termios); err != nil {
		return "", err
	}

	_ = syscall.Kill(0, syscall.SIGSTOP)

	// on resume...
	return "", nil
}
