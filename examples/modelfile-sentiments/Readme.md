# Sentiments Modelfile

This is a simple sentiments analyzer using the Orca model. When you pull Orca from the registry, it has a Template already defined that looks like this:

```Modelfile
### System:
{{ .System }}

### User:
{{ .Prompt }}

### Response:
```

If we just wanted to have the text:

```Plaintext
You are a sentiment analyzer. You will receive text and output only one word, either POSITIVE or NEGATIVE or NEUTRAL, depending on the sentiment of the text.
```

then we could have put this in a SYSTEM block. But we want to provide examples which require updating the full Template. Any Modelfile you create will inherit all the settings from the source model. But in this example, we are overriding the Template.

When providing examples for the input and output, you should include the way the model usually provides information. Since the Orca model expects a user prompt to appear after ### User: and the response is after ### Response, we should format our examples like that as well. If we were using the Llama 2 model, the format would be a bit different.
