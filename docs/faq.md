# FAQ

## How can I upgrade Ollama?

To upgrade Ollama, run the installation process again. On the Mac, click the Ollama icon in the menubar and choose the restart option if an update is available.

## How can I view the logs?

Review the [Troubleshooting](./troubleshooting.md) docs for more about using logs.

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

## What does the q in the model tag mean? What is quantization?

Whenever you pull a model without a tag, Ollama will actually pull the q4_0 quantization of the model. You can verify this on the tags page. On https://ollama.ai/library/llama2/tags you can see that the hash for the latest tag matches the hash for the 7b model. ![quant hashes](https://github.com/jmorganca/ollama/assets/633681/814b1b78-8205-4845-89f9-e671b3b96085)

Looking at the that page for any model, you can see several quantization options available. Quantization is a method of compression that allows the model to fit in less space and thus use less RAM and VRAM on your machine.

At a high level, a model is made of an enormous collection of nodes that determine how to generate text. These nodes are connected at different levels with weights. The training process adjusts these weights to be able to output the right text every time.

Most of the source models that we use start with weights that are 32bit floating-point numbers. Those weights, and another concept called biases, add up to be the parameters. So a source model with 7 billion parameters has 7 billion 32bit floating-point numbers, plus a description of all the nodes and more. That adds up to needing at least 28 Gigabytes of memory to load, if you choose to load one of those source models.

Quantization turns those 32bit floating point weights into much smaller integers. The number next to the q indicates the bit size of the weights. So a q4 model converted those 32bit floats into 4bit integers. A 4bit quantization takes up the space for 7billion 4bit integers, plus a little overhead. That comes out to almost 4 Gigabytes. Obviously, there is some loss of information in this process of going from 30GB to 4GB, but it turns out in most cases it isn't really noticeable. In fact, even the 2bit quantization which fits in less than 3GB can be very useful.

There are three major sets of quantizations you will see in the Ollama Library of models: **fp16**, models with just a q and a number, like **q4_0**, and then models with a **K** in the tag. The **fp16** model is one that has been converted and quantized from the source 32bit to 16bit. This will be about half the size of the 32bit source model and is the largest quantization we deliver in the library. The **q4_0**, **q4_1**, **q5_0**, etc. models use two different quantization methods that were the original methods.

The models with a **K** are often referred to as K Quants. This is a method that allows for models of a similar quality but smaller than the original method used. Essentially, it finds clusters of weights and quantizes those together, allowing for higher precision while using the same bit sizes as the regular quantization options. But this requires a set of maps for the model to figure out the original values which have a computational cost. You may see some impact on the speed of models with K quants compared to the regular quantizations.

## What is context, can I increase it, and why doesn't every model support a huge context?

Context refers to the size of the input you can send to a model and get sensible output back. Many models have a context size of 2048 tokens. It's sometimes possible to give it more using the **num_ctx** parameter, but the answers start to degrade. This is because half of the context is "freed" up to allow for more memory. Newer models have been able to increase that context size using different methods. This increase in context size results in a corresponding increase in memory required, sometimes by orders of magnitude.

> !WARNING]
> Currently, over-allocating context size may result in model quality or stability issues.
