# Public-safety scan: vision image

Scanned image: `dsv4-vision-sm80:path3-final`, pushed as
`ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-vision-sm80-20260902` and
`:latest`. Same procedure as the text-path image
(see `docs/DOCKER-IMAGE.md`), extended with core-IP terms: product repo
names, customer names, internal host/VM codenames, and internal org
names. The internal codename strings are redacted in this receipt (shown
as `CODENAME1`, `CODENAME2`, `ORGNAME`) and were substituted for the real
values when the commands below were actually run.

## 1. Build history

```bash
docker history --no-trunc dsv4-vision-sm80:path3-final > history.log
grep -inE 'hf_[A-Za-z0-9]{20,}|ghp_|gho_|github_pat_|AKIA[0-9A-Z]{12,}|BEGIN (RSA|OPENSSH|PRIVATE)|://[^/[:space:]]*:[^/[:space:]@]*@|100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|seanphan|pixelml|/library|/models/model-cache|CODENAME1|CODENAME2|agent-sandbox|ORGNAME' history.log
```
Result: **0 hits.**

## 2. Full filesystem content scan

```bash
docker run --rm --entrypoint sh dsv4-vision-sm80:path3-final -c '
grep -rlE "hf_[A-Za-z0-9]{20,}|ghp_|gho_|github_pat_|AKIA|BEGIN (RSA|OPENSSH|PRIVATE)|://[^/[:space:]]*:[^/[:space:]@]*@|100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|seanphan|pixelml|/library|/models/model-cache|CODENAME1|CODENAME2|agent-sandbox|ORGNAME" \
  /root /home /etc /workspace /vllm /opt /tmp /var 2>/dev/null'
```
Result: matches only inside vendored, unrelated open-source code —
the CUTLASS submodule and Rust/cargo registry sources, `pip`/`certifi`/
`grpc` CA bundles, Python's own `docs.python.org/3/library` cross-references
in docstrings, the base image's `/etc/resolv.conf` cloud metadata resolver
(`100.100.100.100`, a standard cloud-provider DNS address, not
infrastructure-specific), a generic `ubuntu:...:/home/ubuntu:/bin/bash`
`/etc/passwd` line from the base OS image, and one of the two redacted
codename patterns matching an unrelated upstream optimizer implementation
in Hugging Face `transformers` and a GNU `config.sub` CPU target string —
both pre-existing open-source code, unrelated to this project's internal
naming. None reference PixelML infrastructure, credentials, or private
hosts. **No hit for the other redacted codename, the internal org name,
`agent-sandbox`, or `pixelml`-as-infrastructure-path.**

One vendored `transformers` test-utility file
(`site-packages/transformers/testing_utils.py`) contains a fake token
literal (`hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL`) matching the Hugging Face
token pattern. This is a long-standing public placeholder shipped in
upstream `huggingface/transformers` test code, not a PixelML credential;
reviewed and not treated as a hit.

## 3. Named credential/config file search

```bash
docker run --rm --entrypoint sh dsv4-vision-sm80:path3-final -c '
find /root /home /etc /workspace /vllm /opt /tmp /var -maxdepth 8 \
  \( -iname "*.pem" -o -iname "*.key" -o -iname ".env" -o -iname "hosts.yml" \
     -o -iname "config.json" -o -iname ".netrc" -o -iname ".git-credentials" \
     -o -iname "pip.conf" -o -iname "token" \) 2>/dev/null
find /root /home -iname ".ssh" 2>/dev/null
find / -path "*/.cache/huggingface/token" 2>/dev/null'
```
Result: only public CA bundles (`/etc/ssl/certs/*.pem`, `certifi/cacert.pem`,
`grpc/.../roots.pem`) and unrelated vendored CUTLASS example `config.json`
files. **No private key, `.netrc`, `.git-credentials`, `.ssh` directory, or
cached Hugging Face token.**

## 4. Root shell history and dotfiles

```bash
docker run --rm --entrypoint sh dsv4-vision-sm80:path3-final -c \
  'ls -la /root; cat /root/.bash_history 2>/dev/null | head'
```
Result: `/root` contains only `.bashrc` and `.profile`. **No
`.bash_history`.**

## 5. Model weights / snapshot check

```bash
docker run --rm --entrypoint sh dsv4-vision-sm80:path3-final -c \
  'du -sh /vllm; find / -xdev -iname "*.safetensors" 2>/dev/null'
```
Result: `/vllm` is 6.6 GB of source/build artifacts. Exactly one
`.safetensors` file exists, `compressed_tensors/transform/utils/hadamards.safetensors`,
a small utility tensor shipped inside the `compressed_tensors` pip package,
not model weights. **The image contains no snapshot data.**

## Verdict

Clean. No squashing or layer removal was required before publication.
