# Extern C Server

This directory contains a thin facade we layer on top of the Llama.cpp server to
expose `extern C` interfaces to access the functionality through direct API
calls in-process.  The llama.cpp code uses compile time macros to configure GPU
type along with other settings.  During the `go generate ./...` execution, the
build will generate one or more copies of the llama.cpp `extern C` server based
on what GPU libraries are detected to support multiple GPU types as well as CPU
only support. The Ollama go build then embeds these different servers to support
different GPUs and settings at runtime.

If you are making changes to the code in this directory, make sure to disable
caching during your go build to ensure you pick up your changes.  A typical
iteration cycle from the top of the source tree looks like:

```
go generate ./... && go build -a .
```