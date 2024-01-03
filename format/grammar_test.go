package format

import (
	"bufio"
	"regexp"
	"strings"
	"testing"
)

const objectSchema = `{
    "type": "object",
    "properties": {
        "foo": {
            "type": "number"
        },
        "bar": {
            "type": "string"
        }
    }
}`

const objectGrammar = `
space ::= " "?
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
        )* "\"" space
root ::= "{" space "\"foo\"" space ":" space number "," space "\"bar\"" space ":" space string "}" space
`

const arraySchema = `{
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "firstname": {
                "type": "string"
            },
            "lastname": {
                "type": "string"
            },
            "age": {
                "type": "integer"
            },
            "address": {
                "type": "string"
            },
            "email": {
                "type": "string"
            },
            "isMember": {
                "type": "boolean"
            }
        }
    }
}`

const arrayGrammar = `
space ::= " "?
string ::=  "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
        )* "\"" space
integer ::= ("-"? ([0-9] | [1-9] [0-9]*)) space
boolean ::= ("true" | "false") space
item ::= "{" space "\"address\"" space ":" space string "," space "\"age\"" space ":" space integer "," space "\"email\"" space ":" space string "," space "\"firstname\"" space ":" space string "," space "\"isMember\"" space ":" space boolean "," space "\"lastname\"" space ":" space string "}" space
root ::= "[" space (item ("," space item)*)? "]" space
`

var grammarBeginningRegex = regexp.MustCompile(`^[\w\d\s]+::=`)

func readGrammarRules(grammar string) ([]string, error) {
	scanner := bufio.NewScanner(strings.NewReader(grammar))
	var rules []string
	var rule strings.Builder
	for scanner.Scan() {
		line := scanner.Text()
		if grammarBeginningRegex.MatchString(line) {
			if rule.Len() > 0 {
				rules = append(rules, rule.String())
				rule.Reset()
			}
		}
		rule.WriteString(line)
	}
	if rule.Len() > 0 {
		rules = append(rules, rule.String())
	}
	return rules, scanner.Err()
}

func assertGrammaticallyEqual(t *testing.T, actual, expected string) {
	actualRules, err := readGrammarRules(strings.TrimSpace(actual))
	if err != nil {
		t.Errorf("Error reading actual grammar: %s", err)
		return
	}
	expectedRules, err := readGrammarRules(strings.TrimSpace(expected))
	if err != nil {
		t.Errorf("Error reading expected grammar: %s", err)
		return
	}

	if len(actualRules) != len(expectedRules) {
		t.Errorf("Expected %d rules, got %d", len(expectedRules), len(actualRules))
		return
	}

	expectedSet := make(map[string]bool, len(expectedRules))
	for _, expectedRule := range expectedRules {
		expectedSet[expectedRule] = true
	}
	for _, actualRule := range actualRules {
		if !expectedSet[actualRule] {
			t.Errorf("Unexpected rule found: %s", actualRule)
			return
		}
	}
}

func TestGrammar(t *testing.T) {
	t.Run("object schema to grammar", func(t *testing.T) {
		grammar, err := SchemaToGrammar(objectSchema, []string{"foo", "bar"})
		if err != nil {
			t.Errorf("Error converting schema to grammar: %s", err)
			return
		}
		assertGrammaticallyEqual(t, grammar, objectGrammar)
	})
	t.Run("array schema to grammar", func(t *testing.T) {
		grammar, err := SchemaToGrammar(arraySchema, []string{"address", "age", "email", "firstname", "isMember", "lastname"})
		if err != nil {
			t.Errorf("Error converting schema to grammar: %s", err)
			return
		}
		assertGrammaticallyEqual(t, grammar, arrayGrammar)
	})
}
