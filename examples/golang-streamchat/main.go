package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

func main() {
	err := SendChatRequest(ReqStreamChat{
		Model: ModelLlama27b,
		Messages: []ReqStreamChatMessage{
			{
				Role:    RoleUser,
				Content: "Show me how to use golang channel.",
			},
		},
	})
	if err != nil {
		panic(err)
	}
}

func SendChatRequest(payload ReqStreamChat) error {
	jsonBytes, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("%s: %w", "error marshal payload", err)
	}

	url := "http://localhost:11434/api/chat"
	res, err := http.Post(url, "application/json", bytes.NewBuffer(jsonBytes))
	if err != nil {
		return fmt.Errorf("%s: %w", "error http post request", err)
	}
	defer res.Body.Close()

	dec := json.NewDecoder(res.Body)
	for {
		var r ResStreamChat
		if err := dec.Decode(&r); err != nil {
			break
		}
		fmt.Print(r.Message.Content)
	}

	return nil
}

type Model string

const (
	ModelLlama27b Model = "llama2:7b"
)

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

type ReqStreamChat struct {
	Model    Model                  `json:"model"`
	Messages []ReqStreamChatMessage `json:"messages"`
}

type ReqStreamChatMessage struct {
	Role    Role
	Content string
}

type ResStreamChat struct {
	Model     Model                `json:"model"`
	CreatedAt string               `json:"created_at"`
	Message   ResStreamChatMessage `json:"message"`
	Done      bool                 `json:"done"`
}

type ResStreamChatMessage struct {
	Role    Role        `json:"role"`
	Content string      `json:"content"`
	Images  interface{} `json:"images"`
}
