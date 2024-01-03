// Reference python implementation: https://github.com/ggerganov/llama.cpp/blob/master/examples/json-schema-to-grammar.py

package format

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

const JsonGrammar = `
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
`

var spaceRule = "\" \"?"

var primitiveRules = map[string]string{
	"boolean": `("true" | "false") space`,
	"number":  `("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space`,
	"integer": `("-"? ([0-9] | [1-9] [0-9]*)) space`,
	"string": ` "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
        )* "\"" space`,
	"null": `"null" space`,
}

var invalidRuleCharsRegex = regexp.MustCompile(`[^\dA-Za-z-]+`)

var grammarLiteralExcapes = map[string]string{
	"\n": "\\n",
	"\r": "\\r",
	`"`:  `\"`,
}

type schemaConverter struct {
	propOrder []string
	rules     map[string]string
}

func newSchemaConverter(propOrder []string) *schemaConverter {
	return &schemaConverter{
		propOrder: propOrder,
		rules: map[string]string{
			"space": spaceRule,
		},
	}
}

func (sc *schemaConverter) formatLiteral(literal any) (string, error) {
	literalBytes, err := json.Marshal(literal)
	if err != nil {
		return "", err
	}
	escapedBytes := make([]byte, 0, len(literalBytes)+2)
	escapedBytes = append(escapedBytes, '"')
	for _, b := range literalBytes {
		if escape, ok := grammarLiteralExcapes[string(b)]; ok {
			escapedBytes = append(escapedBytes, []byte(escape)...)
		} else {
			escapedBytes = append(escapedBytes, b)
		}
	}
	escapedBytes = append(escapedBytes, '"')
	return string(escapedBytes), nil
}

func (sc *schemaConverter) addRule(name string, rule string) string {
	key := invalidRuleCharsRegex.ReplaceAllString(name, "-")

	if existingRule, ok := sc.rules[key]; ok {
		if existingRule == rule {
			return key
		}

		for i := 0; ; i++ {
			if _, ok := sc.rules[key+fmt.Sprintf("%d", i)]; !ok {
				key = key + fmt.Sprintf("%d", i)
				break
			}
		}
	}

	sc.rules[key] = rule
	return key
}

func (sc *schemaConverter) visit(schema map[string]interface{}, name string) (string, error) {
	schemaType := schema["type"].(string)
	ruleName := name
	if ruleName == "" {
		ruleName = "root"
	}

	switch {
	case schema["oneOf"] != nil:
		return sc.compileOneOfAnyOf(schema["oneOf"].([]interface{}), ruleName, name)
	case schema["anyOf"] != nil:
		return sc.compileOneOfAnyOf(schema["anyOf"].([]interface{}), ruleName, name)
	case schema["const"] != nil:
		rule, err := sc.formatLiteral(schema["const"])
		if err != nil {
			return "", err
		}
		return sc.addRule(ruleName, rule), nil
	case schema["enum"] != nil:
		return sc.compileEnum(schema["enum"].([]interface{}), ruleName)
	case schemaType == "object" && schema["properties"] != nil:
		return sc.compileObject(schema["properties"].(map[string]interface{}), ruleName, name)
	case schemaType == "array" && schema["items"] != nil:
		return sc.compileArray(schema["items"].(map[string]interface{}), ruleName, name)
	default:
		if primitiveRules[schemaType] == "" {
			return "", fmt.Errorf("unknown schema type: %s", schemaType)
		}
		if ruleName != "root" {
			ruleName = schemaType
		}
		return sc.addRule(ruleName, primitiveRules[schemaType]), nil
	}
}

func (sc *schemaConverter) compileOneOfAnyOf(schemas []interface{}, ruleName, name string) (string, error) {
	var rules = make([]string, len(schemas))
	for i, altSchema := range schemas {
		var key string
		if name == "" {
			key = fmt.Sprintf("%d", i)
		} else {
			key = fmt.Sprintf("%s-%d", name, i)
		}
		rule, err := sc.visit(altSchema.(map[string]interface{}), key)
		if err != nil {
			return "", err
		}
		rules[i] = rule
	}
	return sc.addRule(ruleName, strings.Join(rules, " | ")), nil
}

func (sc *schemaConverter) compileEnum(literals []interface{}, ruleName string) (string, error) {
	var rules = make([]string, len(literals))
	for i, literal := range literals {
		rule, err := sc.formatLiteral(literal)
		if err != nil {
			return "", err
		}
		rules[i] = rule
	}
	return sc.addRule(ruleName, strings.Join(rules, " | ")), nil
}

func (sc *schemaConverter) compileObject(properties map[string]interface{}, ruleName, name string) (string, error) {
	propOrder := sc.propOrder
	propPairs := make([][2]any, 0, len(properties))
	for k, v := range properties {
		propPairs = append(propPairs, [2]any{k, v})
	}
	if len(propOrder) > 0 {
		for i, k := range propOrder {
			for j, pair := range propPairs {
				if pair[0] == k {
					propPairs[i], propPairs[j] = propPairs[j], propPairs[i]
					break
				}
			}
		}
	}

	var rule strings.Builder
	rule.WriteString(`"{" space`)

	i := 0
	for _, propPair := range propPairs {
		propName := propPair[0].(string)
		propSchema := propPair[1].(map[string]interface{})
		var key string
		if name == "" {
			key = propName
		} else {
			key = fmt.Sprintf("%s-%s", name, propName)
		}
		propRuleName, err := sc.visit(propSchema, key)
		if err != nil {
			return "", err
		}
		if i > 0 {
			rule.WriteString(` "," space`)
		}
		propNameLiteral, err := sc.formatLiteral(propName)
		if err != nil {
			return "", err
		}
		fmt.Fprintf(&rule, ` %s space ":" space %s`, propNameLiteral, propRuleName)
		i++
	}

	rule.WriteString(` "}" space`)
	return sc.addRule(ruleName, rule.String()), nil
}

func (sc *schemaConverter) compileArray(items map[string]interface{}, ruleName, name string) (string, error) {
	var key string
	if name == "" {
		key = "item"
	} else {
		key = fmt.Sprintf("%s-item", name)
	}
	itemRuleName, err := sc.visit(items, key)
	if err != nil {
		return "", err
	}
	return sc.addRule(ruleName, fmt.Sprintf(`"[" space (%s ("," space %s)*)? "]" space`, itemRuleName, itemRuleName)), nil
}

func (sc *schemaConverter) formatGrammar() string {
	var b strings.Builder
	for name, rule := range sc.rules {
		fmt.Fprintf(&b, "%s ::= %s\n", name, rule)
	}
	return b.String()
}

// SchemaToGrammar converts a JSON schema to a GNBF grammar,
// with the given property order (optional).
func SchemaToGrammar(schema string, propOrder []string) (string, error) {
	sc := newSchemaConverter(propOrder)
	var schemaObject map[string]interface{}
	err := json.Unmarshal([]byte(schema), &schemaObject)
	if err != nil {
		return "", err
	}
	sc.visit(schemaObject, "")
	return sc.formatGrammar(), nil
}
