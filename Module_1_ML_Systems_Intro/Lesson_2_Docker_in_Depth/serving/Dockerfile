# ---------------------------------------------------------------------------
# Serving image — Node 20 + express + redis client
#
# Structurally identical to training/Dockerfile. Different base image,
# different package manager, different entrypoint. Same pattern.
#
#   deps manifest → install → code → run
# ---------------------------------------------------------------------------

FROM node:20-alpine

WORKDIR /app

# --- Layer 1: dependencies ---------------------------------------------------
# Copied before code so edits to server.js don't re-run npm install.
# `package*.json` catches both package.json and package-lock.json.
COPY package*.json ./

# --production: skip devDependencies. Alpine keeps the image small.
RUN npm install --production

# --- Layer 2: code -----------------------------------------------------------
COPY server.js ./

# Documentation only — Compose still needs `ports:` to publish it.
EXPOSE 3000

CMD ["node", "server.js"]