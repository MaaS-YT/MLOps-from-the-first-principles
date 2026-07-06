FROM node:20-alpine

WORKDIR /app

# Copy package files first — this layer is cached unless dependencies change.
# Changing server.js alone won't invalidate the npm install layer.
COPY package*.json ./
RUN npm install --production

# Copy application code (changes more often than dependencies)
COPY server.js ./

EXPOSE 3000

CMD ["node", "server.js"]
