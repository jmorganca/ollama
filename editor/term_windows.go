package readline

import (
	"syscall"
	"unsafe"
)

const (
	enableLineInput       = 2
	enableWindowInput     = 8
	enableMouseInput      = 16
	enableInsertMode      = 32
	enableQuickEditMode   = 64
	enableExtendedFlags   = 128
	enableProcessedOutput = 1
	enableWrapAtEolOutput = 2
	enableAutoPosition    = 256 // Cursor position is not affected by writing data to the console.
	enableEchoInput       = 4   // Characters are written to the console as they're read.
	enableProcessedInput  = 1   // Enables input processing (like recognizing Ctrl+C).
)

var kernel32 = syscall.NewLazyDLL("kernel32.dll")

var (
	procGetConsoleMode = kernel32.NewProc("GetConsoleMode")
	procSetConsoleMode = kernel32.NewProc("SetConsoleMode")
)

type State struct {
	mode uint32
}

// IsTerminal checks if the given file descriptor is associated with a terminal
func IsTerminal(fd int) bool {
	var st uint32
	r, _, e := syscall.SyscallN(procGetConsoleMode.Addr(), uintptr(fd), uintptr(unsafe.Pointer(&st)), 0)
	// if the call succeeds and doesn't produce an error, it's a terminal
	return r != 0 && e == 0
}

func SetRawMode(fd int) (*State, error) {
	var st uint32
	// retrieve the current mode of the terminal
	_, _, e := syscall.SyscallN(procGetConsoleMode.Addr(), uintptr(fd), uintptr(unsafe.Pointer(&st)), 0)
	if e != 0 {
		return nil, error(e)
	}
	// modify the mode to set it to raw
	raw := st &^ (enableEchoInput | enableProcessedInput | enableLineInput | enableProcessedOutput)
	// apply the new mode to the terminal
	_, _, e = syscall.SyscallN(procSetConsoleMode.Addr(), uintptr(fd), uintptr(raw), 0)
	if e != 0 {
		return nil, error(e)
	}
	// return the original state so that it can be restored later
	return &State{st}, nil
}

func UnsetRawMode(fd int, state *State) error {
	_, _, err := syscall.SyscallN(procSetConsoleMode.Addr(), uintptr(fd), uintptr(state.mode), 0)
	return err
}
