package editor

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"syscall"
)

type Prompt struct {
	Prompt         string
	AltPrompt      string
	Placeholder    string
	AltPlaceholder string
	UseAlt         bool
}

type Terminal struct {
	outchan chan rune
}

type Instance struct {
	Prompt   *Prompt
	Terminal *Terminal
}

func New(prompt Prompt) (*Instance, error) {
	term, err := NewTerminal()
	if err != nil {
		return nil, err
	}

	return &Instance{
		Prompt:   &prompt,
		Terminal: term,
	}, nil
}

func (i *Instance) HandleInput() (string, error) {
	prompt := i.Prompt.Prompt
	if i.Prompt.UseAlt {
		prompt = i.Prompt.AltPrompt
	}
	fmt.Print(prompt)

	termios, err := SetRawMode(syscall.Stdin)
	if err != nil {
		return "", err
	}
	defer UnsetRawMode(syscall.Stdin, termios)

	buf, _ := NewBuffer(i.Prompt)

	var esc bool
	var escex bool
	var pasteMode PasteMode

	fmt.Print(StartBracketedPaste)
	defer fmt.Printf(EndBracketedPaste)

	for {
		if buf.IsEmpty() {
			ph := i.Prompt.Placeholder
			if i.Prompt.UseAlt {
				ph = i.Prompt.AltPlaceholder
			}
			fmt.Printf(ColorGrey + ph + fmt.Sprintf(CursorLeftN, len(ph)) + ColorDefault)
		}

		r, err := i.Terminal.Read()
		if err != nil {
			return "", io.EOF
		}

		if buf.IsEmpty() {
			fmt.Print(ClearToEOL)
		}

		if escex {
			escex = false

			switch r {
			case KeyUp:
				buf.MoveUp()
			case KeyDown:
				buf.MoveDown()
			case KeyLeft:
				buf.MoveLeft()
			case KeyRight:
				buf.MoveRight()
			case CharBracketedPaste:
				var code string
				for cnt := 0; cnt < 3; cnt++ {
					r, err = i.Terminal.Read()
					if err != nil {
						return "", io.EOF
					}

					code += string(r)
				}
				if code == CharBracketedPasteStart {
					pasteMode = PasteModeStart
				} else if code == CharBracketedPasteEnd {
					pasteMode = PasteModeEnd
				}
			case MetaStart:
				buf.MoveToBOL()
			case MetaEnd:
				buf.MoveToEOL()
			}
			continue
		} else if esc {
			esc = false

			switch r {
			case CharEscapeEx:
				escex = true
			}
			continue
		}

		switch r {
		case CharNull:
			continue
		case CharEsc:
			esc = true
		case CharInterrupt:
			return "", ErrInterrupt
		case CharLineStart:
			buf.MoveToBOL()
		case CharLineEnd:
			buf.MoveToEOL()
		case CharBackward:
			buf.MoveLeft()
		case CharForward:
			buf.MoveRight()
		case CharBackspace, CharCtrlH:
			buf.Remove()
		case CharTab:
			for cnt := 0; cnt < 8; cnt++ {
				buf.Add(' ')
			}
		case CharDelete:
			if len(buf.Buf) > 0 && buf.Buf[0].Size() > 0 {
				buf.Delete()
			} else {
				return "", io.EOF
			}
		case CharCtrlU:
			buf.RemoveBefore()
		case CharCtrlL:
			buf.ClearScreen()
		case CharCtrlW:
			buf.RemoveWordBefore()
		case CharCtrlJ:
			buf.Add(r)
		case CharEnter:
			if pasteMode == PasteModeStart {
				buf.Add(r)
				continue
			}
			buf.MoveToEnd()
			fmt.Println()
			return buf.String(), nil
		default:
			if r >= CharSpace || r == CharEnter {
				buf.Add(r)
			}
		}
	}

}

func NewTerminal() (*Terminal, error) {
	t := &Terminal{
		outchan: make(chan rune),
	}

	go t.ioloop()

	return t, nil
}

func (t *Terminal) ioloop() {
	buf := bufio.NewReader(os.Stdin)

	for {
		r, _, err := buf.ReadRune()
		if err != nil {
			close(t.outchan)
			break
		}
		t.outchan <- r
	}
}

func (t *Terminal) Read() (rune, error) {
	r, ok := <-t.outchan
	if !ok {
		return 0, io.EOF
	}

	return r, nil
}
