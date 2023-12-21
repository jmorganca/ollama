package gpu

// Beginning of an `ollama info` command
type GpuInfo struct {
	Driver      string `json:"driver,omitempty"`
	TotalMemory uint64 `json:"total_memory,omitempty"`
	FreeMemory  uint64 `json:"free_memory,omitempty"`

	// TODO add other useful attributes about the card here for discovery information
}
