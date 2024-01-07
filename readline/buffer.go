package readline

import (
	"fmt"
	"os"

	"github.com/emirpasic/gods/lists/arraylist"
	"golang.org/x/term"
)

type Buffer struct {
	Pos       int
	Buf       *arraylist.List
	Prompt    *Prompt
	LineWidth int
	Width     int
	Height    int
}

func NewBuffer(prompt *Prompt) (*Buffer, error) {
	fd := int(os.Stdout.Fd())
	width, height, err := term.GetSize(fd)
	if err != nil {
		fmt.Println("Error getting size:", err)
		return nil, err
	}

	lwidth := width - len(prompt.prompt())

	b := &Buffer{
		Pos:       0,
		Buf:       arraylist.New(),
		Prompt:    prompt,
		Width:     width,
		Height:    height,
		LineWidth: lwidth,
	}

	return b, nil
}

func (b *Buffer) MoveLeft() {
	if b.Pos > 0 {
		if b.Pos%b.LineWidth == 0 {
			fmt.Printf(CursorUp + CursorBOL + cursorRightN(b.Width))
		} else {
			fmt.Print(CursorLeft)
		}
		b.Pos -= 1
	}
}

func (b *Buffer) MoveLeftWord() {
	if b.Pos > 0 {
		var foundNonspace bool
		for {
			v, _ := b.Buf.Get(b.Pos - 1)
			if v == ' ' {
				if foundNonspace {
					break
				}
			} else {
				foundNonspace = true
			}
			b.MoveLeft()

			if b.Pos == 0 {
				break
			}
		}
	}
}

func (b *Buffer) MoveRight() {
	if b.Pos < b.Size() {
		b.Pos += 1
		if b.Pos%b.LineWidth == 0 {
			fmt.Printf(CursorDown + CursorBOL + cursorRightN(len(b.Prompt.prompt())))
		} else {
			fmt.Print(CursorRight)
		}
	}
}

func (b *Buffer) MoveRightWord() {
	if b.Pos < b.Size() {
		for {
			b.MoveRight()
			v, _ := b.Buf.Get(b.Pos)
			if v == ' ' {
				break
			}

			if b.Pos == b.Size() {
				break
			}
		}
	}
}

func (b *Buffer) MoveToStart() {
	if b.Pos > 0 {
		currLine := b.Pos / b.LineWidth
		if currLine > 0 {
			for cnt := 0; cnt < currLine; cnt++ {
				fmt.Print(CursorUp)
			}
		}
		fmt.Printf(CursorBOL + cursorRightN(len(b.Prompt.prompt())))
		b.Pos = 0
	}
}

func (b *Buffer) MoveToEnd() {
	if b.Pos < b.Size() {
		currLine := b.Pos / b.LineWidth
		totalLines := b.Size() / b.LineWidth
		if currLine < totalLines {
			for cnt := 0; cnt < totalLines-currLine; cnt++ {
				fmt.Print(CursorDown)
			}
			remainder := b.Size() % b.LineWidth
			fmt.Printf(CursorBOL + cursorRightN(len(b.Prompt.prompt())+remainder))
		} else {
			fmt.Print(cursorRightN(b.Size() - b.Pos))
		}

		b.Pos = b.Size()
	}
}

func (b *Buffer) Size() int {
	return b.Buf.Size()
}

func min(n, m int) int {
	if n > m {
		return m
	}
	return n
}

func (b *Buffer) Add(r rune) {
	if b.Pos == b.Buf.Size() {
		fmt.Printf("%c", r)
		b.Buf.Add(r)
		b.Pos += 1
		if b.Pos > 0 && b.Pos%b.LineWidth == 0 {
			fmt.Printf("\n%s", b.Prompt.AltPrompt)
		}
	} else {
		fmt.Printf("%c", r)
		b.Buf.Insert(b.Pos, r)
		b.Pos += 1
		if b.Pos > 0 && b.Pos%b.LineWidth == 0 {
			fmt.Printf("\n%s", b.Prompt.AltPrompt)
		}
		b.drawRemaining()
	}
}

func (b *Buffer) drawRemaining() {
	var place int
	remainingText := b.StringN(b.Pos)
	if b.Pos > 0 {
		place = b.Pos % b.LineWidth
	}
	fmt.Print(CursorHide)

	// render the rest of the current line
	currLine := remainingText[:min(b.LineWidth-place, len(remainingText))]
	if len(currLine) > 0 {
		fmt.Printf(ClearToEOL + currLine)
		fmt.Print(cursorLeftN(len(currLine)))
	} else {
		fmt.Print(ClearToEOL)
	}

	// render the other lines
	if len(remainingText) > len(currLine) {
		remaining := []rune(remainingText[len(currLine):])
		var totalLines int
		for i, c := range remaining {
			if i%b.LineWidth == 0 {
				fmt.Printf("\n%s", b.Prompt.AltPrompt)
				totalLines += 1
			}
			fmt.Printf("%c", c)
		}
		fmt.Print(ClearToEOL)
		fmt.Print(cursorUpN(totalLines))
		fmt.Printf(CursorBOL + cursorRightN(b.Width-len(currLine)))
	}

	fmt.Print(CursorShow)
}

func (b *Buffer) Remove() {
	if b.Buf.Size() > 0 && b.Pos > 0 {
		if b.Pos%b.LineWidth == 0 {
			// if the user backspaces over the word boundary, do this magic to clear the line
			// and move to the end of the previous line
			fmt.Printf(CursorBOL + ClearToEOL)
			fmt.Printf(CursorUp + CursorBOL + cursorRightN(b.Width) + " " + CursorLeft)
		} else {
			fmt.Printf(CursorLeft + " " + CursorLeft)
		}

		var eraseExtraLine bool
		if (b.Size()-1)%b.LineWidth == 0 {
			eraseExtraLine = true
		}

		b.Pos -= 1
		b.Buf.Remove(b.Pos)

		if b.Pos < b.Size() {
			b.drawRemaining()
			// this erases a line which is left over when backspacing in the middle of a line and there
			// are trailing characters which go over the line width boundary
			if eraseExtraLine {
				remainingLines := (b.Size() - b.Pos) / b.LineWidth
				fmt.Printf(cursorDownN(remainingLines+1) + CursorBOL + ClearToEOL)
				place := b.Pos % b.LineWidth
				fmt.Printf(cursorUpN(remainingLines+1) + cursorRightN(place+len(b.Prompt.prompt())))
			}
		}
	}
}

func (b *Buffer) Delete() {
	if b.Size() > 0 && b.Pos < b.Size() {
		b.Buf.Remove(b.Pos)
		b.drawRemaining()
		if b.Size()%b.LineWidth == 0 {
			if b.Pos != b.Size() {
				remainingLines := (b.Size() - b.Pos) / b.LineWidth
				fmt.Printf(cursorDownN(remainingLines) + CursorBOL + ClearToEOL)
				place := b.Pos % b.LineWidth
				fmt.Printf(cursorUpN(remainingLines) + cursorRightN(place+len(b.Prompt.prompt())))
			}
		}
	}
}

func (b *Buffer) DeleteBefore() {
	if b.Pos > 0 {
		for cnt := b.Pos - 1; cnt >= 0; cnt-- {
			b.Remove()
		}
	}
}

func (b *Buffer) DeleteRemaining() {
	if b.Size() > 0 && b.Pos < b.Size() {
		charsToDel := b.Size() - b.Pos
		for cnt := 0; cnt < charsToDel; cnt++ {
			b.Delete()
		}
	}
}

func (b *Buffer) DeleteWord() {
	if b.Buf.Size() > 0 && b.Pos > 0 {
		var foundNonspace bool
		for {
			v, _ := b.Buf.Get(b.Pos - 1)
			if v == ' ' {
				if !foundNonspace {
					b.Remove()
				} else {
					break
				}
			} else {
				foundNonspace = true
				b.Remove()
			}

			if b.Pos == 0 {
				break
			}
		}
	}
}

func (b *Buffer) ClearScreen() {
	fmt.Printf(ClearScreen + CursorReset + b.Prompt.prompt())
	if b.IsEmpty() {
		ph := b.Prompt.placeholder()
		fmt.Printf(ColorGrey + ph + cursorLeftN(len(ph)) + ColorDefault)
	} else {
		currPos := b.Pos
		b.Pos = 0
		b.drawRemaining()
		fmt.Printf(CursorReset + cursorRightN(len(b.Prompt.prompt())))
		if currPos > 0 {
			targetLine := currPos / b.LineWidth
			if targetLine > 0 {
				for cnt := 0; cnt < targetLine; cnt++ {
					fmt.Print(CursorDown)
				}
			}
			remainder := currPos % b.LineWidth
			if remainder > 0 {
				fmt.Print(cursorRightN(remainder))
			}
			if currPos%b.LineWidth == 0 {
				fmt.Printf(CursorBOL + b.Prompt.AltPrompt)
			}
		}
		b.Pos = currPos
	}
}

func (b *Buffer) IsEmpty() bool {
	return b.Buf.Empty()
}

func (b *Buffer) Replace(r []rune) {
	b.Pos = 0
	b.Buf.Clear()
	fmt.Printf(ClearLine + CursorBOL + b.Prompt.prompt())
	for _, c := range r {
		b.Add(c)
	}
}

func (b *Buffer) String() string {
	return b.StringN(0)
}

func (b *Buffer) StringN(n int) string {
	return b.StringNM(n, 0)
}

func (b *Buffer) StringNM(n, m int) string {
	var s string
	if m == 0 {
		m = b.Size()
	}
	for cnt := n; cnt < m; cnt++ {
		c, _ := b.Buf.Get(cnt)
		s += string(c.(rune))
	}
	return s
}

func cursorLeftN(n int) string {
	return fmt.Sprintf(CursorLeftN, n)
}

func cursorRightN(n int) string {
	return fmt.Sprintf(CursorRightN, n)
}

func cursorUpN(n int) string {
	return fmt.Sprintf(CursorUpN, n)
}

func cursorDownN(n int) string {
	return fmt.Sprintf(CursorDownN, n)
}
