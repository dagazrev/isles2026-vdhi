FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS isles26_baseline
# GC T4 note: organizer template prefers cuda12.6. This 12.4 runtime still
# targets sm_75 (T4). Local Blackwell smoke uses CPU fallback in inference.py.
# Swap to pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime if GC rejects 12.4.

ENV PYTHONUNBUFFERED=1
# GC limit: ~7 minutes per case on a T4 — keep inference lean (see inference.py).
ENV ISLES_MAX_CASE_SECONDS=420

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user app.py /opt/app/
COPY --chown=user:user inference.py /opt/app/

LABEL org.grand-challenge.api-method="invoke"
EXPOSE 4743
ENTRYPOINT ["python", "app.py"]
