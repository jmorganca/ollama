package editor

import (
	"fmt"
	"strings"

	"github.com/emirpasic/gods/lists/arraylist"
	"golang.org/x/term"
)

type Buffer struct {
	PosX         int
	PosY         int
	Buf          []*arraylist.List
	Prompt       *Prompt
	WordWrap     int
	ScreenWidth  int
	ScreenHeight int
}

func NewBuffer(prompt *Prompt) (*Buffer, error) {
	width, height, err := term.GetSize(0)
	if err != nil {
		fmt.Println("Error getting size:", err)
		return nil, err
	}

	b := &Buffer{
		PosX:         0,
		PosY:         0,
		Buf:          []*arraylist.List{arraylist.New()},
		Prompt:       prompt,
		ScreenWidth:  width,
		ScreenHeight: height,
	}

	return b, nil
}

func (b *Buffer) LineWidth() int {
	return b.ScreenWidth - len(b.Prompt.Prompt)
}

func (b *Buffer) findWordAtPos(line string, pos int) string {
	return ""
}

func (b *Buffer) addLine(row int) {
	if row+1 == len(b.Buf) {
		b.Buf = append(b.Buf, arraylist.New())
	} else {
		b.Buf = append(b.Buf, nil)
		copy(b.Buf[row+2:], b.Buf[row+1:])
		b.Buf[row+1] = arraylist.New()
	}
}

func (b *Buffer) Add(r rune) {
	switch r {
	case CharCtrlJ, CharEnter:
		b.addLine(b.PosY)

		// handle Ctrl-J in the middle of a line
		var remainingText string
		if b.PosX < b.Buf[b.PosY].Size() {
			fmt.Print(ClearToEOL)
			remainingText = b.StringLine(b.PosX, b.PosY)
			for cnt := 0; cnt < len(remainingText); cnt++ {
				b.Buf[b.PosY].Remove(b.Buf[b.PosY].Size() - 1)
				b.Buf[b.PosY+1].Add(rune(remainingText[cnt]))
			}
		}
		b.PosY++
		b.PosX = 0
		fmt.Printf("\n... " + ClearToEOL)
		b.drawRemaining()
	default:
		if b.PosX == b.Buf[b.PosY].Size() {
			fmt.Printf("%c", r)
			b.PosX++
			b.Buf[b.PosY].Add(r)
			wrap, prefix, offset := b.splitLineInsert(b.PosY, b.PosX)
			if wrap {
				fmt.Print(CursorHide + cursorLeftN(len(prefix)+1) + ClearToEOL)
				fmt.Printf("\n%s... %s%c", ClearToEOL, prefix, r)
				b.PosY++
				b.PosX = offset
				b.ResetCursor()
				b.drawRemaining()
				fmt.Print(CursorShow)
			}
		} else {
			fmt.Printf("%c", r)
			b.Buf[b.PosY].Insert(b.PosX, r)
			b.PosX++
			_, prefix, offset := b.splitLineInsert(b.PosY, b.PosX)
			fmt.Print(CursorHide)
			if b.PosX > b.Buf[b.PosY].Size() {
				if offset > 0 {
					fmt.Print(cursorLeftN(offset))
				}
				fmt.Print(ClearToEOL + CursorDown + CursorBOL + ClearToEOL)
				fmt.Printf("... %s", prefix[:offset])
				b.PosY++
				b.PosX = offset
				b.ResetCursor()
			}
			b.drawRemaining()
			fmt.Print(CursorShow)
		}
	}
}

func (b *Buffer) ResetCursor() {
	fmt.Print(CursorHide + CursorBOL)
	fmt.Print(cursorRightN(b.PosX + len(b.Prompt.Prompt)))
	fmt.Print(CursorShow)
}

func (b *Buffer) splitLineInsert(posY, posX int) (bool, string, int) {
	line := b.StringLine(0, posY)
	screenEdge := b.LineWidth() - 5

	// if the current line doesn't need to be reflowed, none of the other
	// lines will either
	if len(line) <= screenEdge {
		return false, "", 0
	}

	// we know we're going to have to insert onto the next line, so
	// add another line if there isn't one already
	if posY == len(b.Buf)-1 {
		b.Buf = append(b.Buf, arraylist.New())
	}

	// make a truncated version of the current line
	currLine := line[:screenEdge]

	// figure out where the last space in the line is
	idx := strings.LastIndex(currLine, " ")

	// deal with strings that don't have spaces in them
	if idx == -1 {
		idx = len(currLine) - 1
	}

	// if the next line already has text on it, we need
	// to add a space to insert our new word
	if b.Buf[posY+1].Size() > 0 {
		b.Buf[posY+1].Insert(0, ' ')
	}

	// calculate the number of characters we need to remove
	// from the current line to add to the next one
	totalChars := len(line) - idx - 1

	for cnt := 0; cnt < totalChars; cnt++ {
		b.Buf[posY].Remove(b.Buf[posY].Size() - 1)
		b.Buf[posY+1].Insert(0, rune(line[len(line)-1-cnt]))
	}
	// remove the trailing space
	b.Buf[posY].Remove(b.Buf[posY].Size() - 1)

	// wrap any further lines
	if b.Buf[posY+1].Size() > b.LineWidth()-5 {
		b.splitLineInsert(posY+1, 0)
	}

	return true, currLine[idx+1:], posX - idx - 1
}

func (b *Buffer) drawRemaining() {
	remainingText := b.StringFromRow(b.PosY)
	remainingText = remainingText[b.PosX:]

	fmt.Print(CursorHide + ClearToEOL)

	var rowCount int
	for _, c := range remainingText {
		fmt.Print(string(c))
		if c == '\n' {
			fmt.Print("... " + ClearToEOL)
			rowCount++
		}
	}
	if rowCount > 0 {
		fmt.Print(cursorUpN(rowCount))
	}
	b.ResetCursor()
}

func (b *Buffer) findWordBeginning(posX int) int {
	for {
		if posX < 0 {
			return -1
		}
		r, ok := b.Buf[b.PosY].Get(posX)
		if !ok {
			return -1
		} else if r.(rune) == ' ' {
			return posX
		}
		posX--
	}
}

func (b *Buffer) Delete() {
	if b.PosX < b.Buf[b.PosY].Size()-1 {
		b.Buf[b.PosY].Remove(b.PosX)
		b.drawRemaining()
	} else {
		b.joinLines()
	}
}

func (b *Buffer) joinLines() {
	lineLen := b.Buf[b.PosY].Size()
	for cnt := 0; cnt < lineLen; cnt++ {
		r, _ := b.Buf[b.PosY].Get(0)
		b.Buf[b.PosY].Remove(0)
		b.Buf[b.PosY-1].Add(r)
	}
}

func (b *Buffer) Remove() {
	if b.PosX > 0 {
		fmt.Print(CursorLeft + " " + CursorLeft)
		b.PosX--
		b.Buf[b.PosY].Remove(b.PosX)
		if b.PosX < b.Buf[b.PosY].Size() {
			fmt.Print(ClearToEOL)
			b.drawRemaining()
		}
	} else if b.PosX == 0 && b.PosY > 0 {
		b.joinLines()

		lastPos := b.Buf[b.PosY-1].Size()
		var cnt int
		b.PosX = lastPos
		b.PosY--

		fmt.Print(CursorHide)
		for {
			if b.PosX+cnt > b.LineWidth()-5 {
				// the concatenated line won't fit, so find the beginning of the word
				// and copy the rest of the string from there
				idx := b.findWordBeginning(b.PosX)
				lineLen := b.Buf[b.PosY].Size()
				for offset := idx + 1; offset < lineLen; offset++ {
					r, _ := b.Buf[b.PosY].Get(idx + 1)
					b.Buf[b.PosY].Remove(idx + 1)
					b.Buf[b.PosY+1].Add(r)
				}
				// remove the trailing space
				b.Buf[b.PosY].Remove(idx)
				fmt.Print(CursorUp + ClearToEOL)
				b.PosX = 0
				b.drawRemaining()
				fmt.Print(CursorDown)
				if idx > 0 {
					if lastPos-idx-1 > 0 {
						b.PosX = lastPos - idx - 1
						b.ResetCursor()
					}
				}
				b.PosY++
				break
			}
			r, ok := b.Buf[b.PosY].Get(b.PosX + cnt)
			if !ok {
				// found the end of the string
				fmt.Print(CursorUp + cursorRightN(b.PosX) + ClearToEOL)
				b.drawRemaining()
				break
			}
			if r == ' ' {
				// found the end of the word
				lineLen := b.Buf[b.PosY].Size()
				for offset := b.PosX + cnt + 1; offset < lineLen; offset++ {
					r, _ := b.Buf[b.PosY].Get(b.PosX + cnt + 1)
					b.Buf[b.PosY].Remove(b.PosX + cnt + 1)
					b.Buf[b.PosY+1].Add(r)
				}
				fmt.Print(CursorUp + cursorRightN(b.PosX) + ClearToEOL)
				b.drawRemaining()
				break
			}
			cnt++
		}
		fmt.Print(CursorShow)
	}
}

func (b *Buffer) RemoveBefore() {
	for {
		if b.PosX == 0 && b.PosY == 0 {
			break
		}
		b.Remove()
	}
}

func (b *Buffer) RemoveWordBefore() {
	if b.PosX > 0 || b.PosY > 0 {
		var foundNonspace bool
		for {
			xPos := b.PosX
			yPos := b.PosY

			v, _ := b.Buf[yPos].Get(xPos - 1)
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

			if xPos == 0 && yPos == 0 {
				break
			}
		}
	}
}

func (b *Buffer) StringLine(x, y int) string {
	if y >= len(b.Buf) {
		return ""
	}

	var output string

	for cnt := x; cnt < b.Buf[y].Size(); cnt++ {
		r, _ := b.Buf[y].Get(cnt)
		output += string(r.(rune))
	}
	return output
}

func (b *Buffer) String() string {
	return b.StringFromRow(0)
}

func (b *Buffer) StringFromRow(n int) string {
	var output []string
	for _, row := range b.Buf[n:] {
		var currLine string
		for cnt := 0; cnt < row.Size(); cnt++ {
			r, _ := row.Get(cnt)
			currLine += string(r.(rune))
		}
		currLine = strings.TrimRight(currLine, " ")
		output = append(output, currLine)
	}
	return strings.Join(output, "\n")
}

func (b *Buffer) cursorUp() {
	fmt.Print(CursorUp)
	b.ResetCursor()
}

func (b *Buffer) cursorDown() {
	fmt.Print(CursorDown)
	b.ResetCursor()
}

func (b *Buffer) MoveUp() {
	if b.PosY > 0 {
		b.PosY--
		if b.Buf[b.PosY].Size() < b.PosX {
			b.PosX = b.Buf[b.PosY].Size()
		}
		b.cursorUp()
	} else {
		fmt.Print("\a")
	}
}

func (b *Buffer) MoveDown() {
	if b.PosY < len(b.Buf)-1 {
		b.PosY++
		if b.Buf[b.PosY].Size() < b.PosX {
			b.PosX = b.Buf[b.PosY].Size()
		}
		b.cursorDown()
	} else {
		fmt.Print("\a")
	}
}

func (b *Buffer) MoveLeft() {
	if b.PosX > 0 {
		b.PosX--
		fmt.Print(CursorLeft)
	} else if b.PosY > 0 {
		b.PosX = b.Buf[b.PosY-1].Size()
		b.PosY--
		b.cursorUp()
	} else if b.PosX == 0 && b.PosY == 0 {
		fmt.Print("\a")
	}
}

func (b *Buffer) MoveRight() {
	if b.PosX < b.Buf[b.PosY].Size() {
		b.PosX++
		fmt.Print(CursorRight)
	} else if b.PosY < len(b.Buf)-1 {
		b.PosY++
		b.PosX = 0
		b.cursorDown()
	} else {
		fmt.Print("\a")
	}
}

func (b *Buffer) MoveToBOL() {
	if b.PosX > 0 {
		b.PosX = 0
		b.ResetCursor()
	}
}

func (b *Buffer) MoveToEOL() {
	if b.PosX < b.Buf[b.PosY].Size() {
		b.PosX = b.Buf[b.PosY].Size()
		b.ResetCursor()
	}
}

func (b *Buffer) MoveToEnd() {
	fmt.Print(CursorHide)
	yDiff := len(b.Buf)-1 - b.PosY
	if yDiff > 0 {
		fmt.Print(cursorDownN(yDiff))
	}
	b.PosY = len(b.Buf)-1
	b.MoveToEOL()
	fmt.Print(CursorShow)
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

func (b *Buffer) ClearScreen() {
	fmt.Printf(CursorHide + ClearScreen + CursorReset + b.Prompt.Prompt)
	if b.IsEmpty() {
		ph := b.Prompt.Placeholder
		fmt.Printf(ColorGrey + ph + cursorLeftN(len(ph)) + ColorDefault)
	} else {
		currPosX := b.PosX
		currPosY := b.PosY
		b.PosX = 0
		b.PosY = 0
		b.drawRemaining()
		b.PosX = currPosX
		b.PosY = currPosY
		fmt.Print(CursorReset + cursorRightN(len(b.Prompt.Prompt)))
		if b.PosY > 0 {
			fmt.Print(cursorDownN(b.PosY))
		}
		if b.PosX > 0 {
			fmt.Print(cursorRightN(b.PosX))
		}
	}
	fmt.Print(CursorShow)
}

func (b *Buffer) IsEmpty() bool {
	return len(b.Buf) == 1 && b.Buf[0].Empty()
}
