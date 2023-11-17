package editor

const (
	CharNull      = 0
	CharLineStart = 1
	CharBackward  = 2
	CharInterrupt = 3
	CharDelete    = 4
	CharLineEnd   = 5
	CharForward   = 6
	CharBell      = 7
	CharCtrlH     = 8
	CharTab       = 9
	CharCtrlJ     = 10
	CharKill      = 11
	CharCtrlL     = 12
	CharEnter     = 13
	CharNext      = 14
	CharPrev      = 16
	CharBckSearch = 18
	CharFwdSearch = 19
	CharTranspose = 20
	CharCtrlU     = 21
	CharCtrlW     = 23
	CharCtrlY     = 25
	CharCtrlZ     = 26
	CharEsc       = 27
	CharSpace     = 32
	CharEscapeEx  = 91
	CharBackspace = 127
)

const (
	KeyDel    = 51
	KeyUp     = 65
	KeyDown   = 66
	KeyRight  = 67
	KeyLeft   = 68
	MetaEnd   = 70
	MetaStart = 72
)

const (
	CursorUp    = "\033[1A"
	CursorDown  = "\033[1B"
	CursorRight = "\033[1C"
	CursorLeft  = "\033[1D"

	CursorSave    = "\033[s"
	CursorRestore = "\033[u"

	CursorUpN    = "\033[%dA"
	CursorDownN  = "\033[%dB"
	CursorRightN = "\033[%dC"
	CursorLeftN  = "\033[%dD"

	CursorEOL  = "\033[E"
	CursorBOL  = "\033[1G"
	CursorHide = "\033[?25l"
	CursorShow = "\033[?25h"

	ClearToEOL  = "\033[K"
	ClearLine   = "\033[2K"
	ClearScreen = "\033[2J"
	CursorReset = "\033[0;0f"

	ColorGrey    = "\033[38;5;245m"
	ColorDefault = "\033[0m"

	StartBracketedPaste = "\033[?2004h"
	EndBracketedPaste   = "\033[?2004l"
)

const (
	CharBracketedPaste      = 50
	CharBracketedPasteStart = "00~"
	CharBracketedPasteEnd   = "01~"
)

type PasteMode int

const (
	PastModeOff = iota
	PasteModeStart
	PasteModeEnd
)
