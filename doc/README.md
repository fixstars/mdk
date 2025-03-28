# MDK Documentation

## Installation

```bash
sudo apt install npm
npm i

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
nvm install --lts
source ~/.bashrc
```

### Local Development

```bash
npm run start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

```bash
cd build
aws s3 sync . s3://mdk-doc-cdk --profile=<aws profile>
```
