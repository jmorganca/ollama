# FAQ

## How can I upgrade Ollama?

To upgrade Ollama, run the installation process again. On the Mac, click the Ollama icon in the menubar and choose the restart option if an update is available.

## How can I view the logs?

Review the [Troubleshooting](./troubleshooting.md) docs for more about using logs.

## What are the components of Ollama that need to be running to work with the CLI, the API, and 3rd party tools?

At the heart of Ollama there are two main components: the server and the client. Even if everything is running on a single machine, there is a server that is running as a service, or background process, and there is some sort of client. Often that is the CLI. For instance, `ollama run llama2` is the command to start the CLI. You will often see this referred to as the REPL, a tool where you can interactively work with Ollama. You can run the server using the command `ollama serve`, but we recommend letting the service run instead. The Ollama installer script for Linux will add a systemd service to your machine that runs `ollama serve` as the user, `ollama`. On macOS, running `ollama` will start the Ollama Menu Bar app which is running the service.

The Ollama service is what actually loads the model and processes your requests. It also serves the API that all clients use, including our REPL and any 3rd party tools. There are some tools that require adding some environment variables to make the service more accessible in different ways. You can learn more about configuring those below.

## How do I use Ollama server environment variables on Mac

On macOS, Ollama runs in the background and is managed by the menubar app. If adding environment variables, Ollama will need to be run manually.

1. Click the menubar icon for Ollama and choose **Quit Ollama**.
2. Open a new terminal window and run the following command (this example uses `OLLAMA_HOST` with an IP address of `123.1.1.1`):

   ```bash
   OLLAMA_HOST=123.1.1.1 ollama serve
   ```

## How do I use Ollama server environment variables on Linux?

If Ollama is installed with the install script, a systemd service was created, running as the Ollama user. To add an environment variable, such as OLLAMA_HOST, follow these steps:

1. Create a `systemd` drop-in directory and add a config file. This is only needed once.

   ```bash
   mkdir -p /etc/systemd/system/ollama.service.d
   echo '[Service]' >>/etc/systemd/system/ollama.service.d/environment.conf
   ```

2. For each environment variable, add it to the config file:

   ```bash
   echo 'Environment="OLLAMA_HOST=0.0.0.0:11434"' >>/etc/systemd/system/ollama.service.d/environment.conf
   ```

3. Reload `systemd` and restart Ollama:

   ```bash
   systemctl daemon-reload
   systemctl restart ollama
   ```

## How can I expose Ollama on my network?

Ollama binds to 127.0.0.1 port 11434 by default. Change the bind address with the `OLLAMA_HOST` environment variable. Refer to the section above for how to use environment variables on your platform.

## How can I allow additional web origins to access Ollama?

Ollama allows cross-origin requests from `127.0.0.1` and `0.0.0.0` by default. Add additional origins with the `OLLAMA_ORIGINS` environment variable. For example, to add all ports on 192.168.1.1 and https://example.com, use:

```shell
OLLAMA_ORIGINS=http://192.168.1.1:*,https://example.com
```

Refer to the section above for how to use environment variables on your platform.

## Where are models stored?

- macOS: `~/.ollama/models`.
- Linux: `/usr/share/ollama/.ollama/models`

## How do I set them to a different location?

If a different directory needs to be used, set the environment variable `OLLAMA_MODELS` to the chosen directory. Refer to the section above for how to use environment variables on your platform.

## Does Ollama send my prompts and answers back to Ollama.ai to use in any way?

No, Ollama runs entirely locally, and conversation data will never leave your machine.

## How can I use Ollama in Visual Studio Code?

There is already a large collection of plugins available for VSCode as well as other editors that leverage Ollama. See the list of [extensions & plugins](https://github.com/jmorganca/ollama#extensions--plugins) at the bottom of the main repository readme.

## How do I use Ollama behind a proxy?

Ollama is compatible with proxy servers if `HTTP_PROXY` or `HTTPS_PROXY` are configured. When using either variables, ensure it is set where `ollama serve` can access the values. When using `HTTPS_PROXY`, ensure the proxy certificate is installed as a system certificate. Refer to the section above for how to use environment variables on your platform.

### How do I use Ollama behind a proxy in Docker?

The Ollama Docker container image can be configured to use a proxy by passing `-e HTTPS_PROXY=https://proxy.example.com` when starting the container.

Alternatively, the Docker daemon can be configured to use a proxy. Instructions are available for Docker Desktop on [macOS](https://docs.docker.com/desktop/settings/mac/#proxies), [Windows](https://docs.docker.com/desktop/settings/windows/#proxies), and [Linux](https://docs.docker.com/desktop/settings/linux/#proxies), and Docker [daemon with systemd](https://docs.docker.com/config/daemon/systemd/#httphttps-proxy).

Ensure the certificate is installed as a system certificate when using HTTPS. This may require a new Docker image when using a self-signed certificate.

```dockerfile
FROM ollama/ollama
COPY my-ca.pem /usr/local/share/ca-certificates/my-ca.crt
RUN update-ca-certificates
```

Build and run this image:

```shell
docker build -t ollama-with-ca .
docker run -d -e HTTPS_PROXY=https://my.proxy.example.com -p 11434:11434 ollama-with-ca
```

## How do I use Ollama with GPU acceleration in Docker?

The Ollama Docker container can be configured with GPU acceleration in Linux or Windows (with WSL2). This requires the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit). See [ollama/ollama](https://hub.docker.com/r/ollama/ollama) for more details.

GPU acceleration is not available for Docker Desktop in macOS due to the lack of GPU passthrough and emulation.

## Why is networking slow in WSL2 on Windows 10?

This can impact both installing Ollama, as well as downloading models.

Open `Control Panel > Networking and Internet > View network status and tasks` and click on `Change adapter settings` on the left panel. Find the `vEthernel (WSL)` adapter, right click and select `Properties`.
Click on `Configure` and open the `Advanced` tab. Search through each of the properties until you find `Large Send Offload Version 2 (IPv4)` and `Large Send Offload Version 2 (IPv6)`. *Disable* both of these
properties.
