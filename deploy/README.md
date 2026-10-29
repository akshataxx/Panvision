# Deploying PanVision to a Compute Engine VM

```text
GitHub push to main
   └─ Actions: tests → build image → push to Artifact Registry → SSH (via IAP) → docker compose up

Browser ──HTTPS:443──▶ caddy ──▶ web (Streamlit :8501) ──http://api:8001──▶ api (FastAPI) ──TLS:5432──▶ Neon
                       (VM, Docker network; only 80/443 are public)        └── /models (read-only weights)
```

- **caddy** terminates HTTPS (automatic Let's Encrypt) and is the only public entry point.
- **web** calls **api** server-side over the private Docker network using the service name `api`.
  `PANVISION_INTERNAL_API_HOSTS=api` allows that one hostname over plain HTTP.
- **api** is the only service with the database URL. It reaches Neon over the internet with TLS
  (`sslmode=require`). The VM and Neon are both in Singapore (`asia-southeast1` / `aws-ap-southeast-1`).
- Secrets (`.env.prod`) and model weights live only on the VM, never in GitHub or the image.

Files on the VM in `/opt/panvision`:

| File | Who writes it |
| --- | --- |
| `docker-compose.yml`, `Caddyfile` | The deploy workflow, on every push |
| `.env` (image tag + domain) | The deploy workflow, on every push |
| `.env.prod` (`PANVISION_DATABASE_URL`) | You, once (`chmod 600`) |
| `models/` (P3, L2 and `densenet121_predicted` classifier) | You, once and when the models change |

## One-time setup

Steps 1–3 are also bundled as `deploy/setup-gcp.sh`, which can be pasted into Google Cloud Shell
(no local installs needed); it prints the step 4 values at the end. To run the steps by hand from a
laptop with `gcloud` signed in, adjust the first line below.

```bash
PROJECT=your-gcp-project  REGION=asia-southeast1  ZONE=asia-southeast1-a  VM=panvision  REPO=hrihaanG/Panvision
gcloud config set project $PROJECT
PROJECT_NUMBER=$(gcloud projects describe $PROJECT --format='value(projectNumber)')

gcloud services enable compute.googleapis.com artifactregistry.googleapis.com \
  iamcredentials.googleapis.com iap.googleapis.com
gcloud artifacts repositories create panvision --repository-format=docker --location=$REGION
```

### 1. VM

```bash
gcloud iam service-accounts create panvision-vm
VM_SA=panvision-vm@$PROJECT.iam.gserviceaccount.com
gcloud artifacts repositories add-iam-policy-binding panvision --location=$REGION \
  --member=serviceAccount:$VM_SA --role=roles/artifactregistry.reader

gcloud compute addresses create panvision-ip --region=$REGION
# e2-standard-2 (8 GB) handles typical scans; resize to e2-standard-4 if segmentation runs out of memory.
gcloud compute instances create $VM --zone=$ZONE --machine-type=e2-standard-2 \
  --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud --boot-disk-size=50GB \
  --service-account=$VM_SA --scopes=cloud-platform --address=panvision-ip \
  --metadata=enable-oslogin=TRUE --tags=panvision

gcloud compute firewall-rules create panvision-web --allow=tcp:80,tcp:443 --target-tags=panvision
# SSH only through Identity-Aware Proxy (Google's fixed range), not the open internet.
gcloud compute firewall-rules create panvision-iap-ssh --allow=tcp:22 \
  --source-ranges=35.235.240.0/20 --target-tags=panvision
# Optional hardening: gcloud compute firewall-rules delete default-allow-ssh
```

Point a DNS **A record** for your domain at the reserved IP
(`gcloud compute addresses describe panvision-ip --region=$REGION --format='value(address)'`).

### 2. Docker, weights and secrets on the VM

```bash
gcloud compute ssh $VM --zone=$ZONE --tunnel-through-iap --command '
  curl -fsSL https://get.docker.com | sudo sh
  sudo gcloud auth configure-docker '$REGION'-docker.pkg.dev --quiet
  sudo install -d /opt/panvision/models'

gcloud compute scp --recurse data/private/models/P3_nnunet_2d_full data/private/models/L2_attention_unet_roi_full \
  $VM:~/ --zone=$ZONE --tunnel-through-iap
gcloud compute ssh $VM --zone=$ZONE --tunnel-through-iap --command '
  sudo mv ~/P3_nnunet_2d_full ~/L2_attention_unet_roi_full /opt/panvision/models/
  sudo chmod -R a+rX /opt/panvision/models'
```

Then SSH in (`gcloud compute ssh $VM --zone=$ZONE --tunnel-through-iap`) and create the secrets file:

```bash
sudo tee /opt/panvision/.env.prod > /dev/null <<'EOF'
PANVISION_DATABASE_URL='postgresql://RUNTIME_USER:PASSWORD@HOST-pooler.ap-southeast-1.aws.neon.tech/DATABASE?sslmode=require'
EOF
sudo chmod 600 /opt/panvision/.env.prod
```

Use Neon's **pooled** connection string and, ideally, a runtime role without schema-creation rights.

### 3. GitHub → Google Cloud trust (no stored keys)

```bash
gcloud iam service-accounts create github-deployer
DEPLOY_SA=github-deployer@$PROJECT.iam.gserviceaccount.com

gcloud artifacts repositories add-iam-policy-binding panvision --location=$REGION \
  --member=serviceAccount:$DEPLOY_SA --role=roles/artifactregistry.writer
for role in roles/compute.viewer roles/compute.osAdminLogin roles/iap.tunnelResourceAccessor; do
  gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:$DEPLOY_SA --role=$role
done
gcloud iam service-accounts add-iam-policy-binding $VM_SA \
  --member=serviceAccount:$DEPLOY_SA --role=roles/iam.serviceAccountUser

gcloud iam workload-identity-pools create github --location=global
gcloud iam workload-identity-pools providers create-oidc github --location=global \
  --workload-identity-pool=github --issuer-uri=https://token.actions.githubusercontent.com \
  --attribute-mapping=google.subject=assertion.sub,attribute.repository=assertion.repository \
  --attribute-condition="assertion.repository=='$REPO'"
gcloud iam service-accounts add-iam-policy-binding $DEPLOY_SA --role=roles/iam.workloadIdentityUser \
  --member=principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/$REPO
```

### 4. GitHub repository variables

None of these are secret. A repository admin sets them (Settings → Secrets and variables → Actions → Variables, or `gh`):

```bash
gh variable set GCP_PROJECT_ID   --body $PROJECT
gh variable set GCP_REGION       --body $REGION
gh variable set GCP_ZONE         --body $ZONE
gh variable set GCP_VM_NAME      --body $VM
gh variable set GCP_DEPLOY_SA    --body $DEPLOY_SA
gh variable set GCP_WIF_PROVIDER --body projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github
gh variable set PANVISION_DOMAIN --body panvision.example.com
```

The workflow deploys to a GitHub environment called `production`. Add required reviewers there if you want a manual approval gate.

### 5. First deploy and database setup

Push to `main` (or run the workflow manually from the Actions tab). After it succeeds, initialise the
schema and the first administrator once, using an owner-role URL:

```bash
cd /opt/panvision
sudo docker compose run --rm -e PANVISION_DATABASE_URL='OWNER_ROLE_URL' api python -m app.database init
sudo docker compose run --rm -e PANVISION_DATABASE_URL='OWNER_ROLE_URL' api python -m app.database create-admin
```

Schema changes are not applied automatically on deploy. Re-run `init` (additive and idempotent) when a
release adds tables or columns.

## Day to day

- **Deploy:** push or merge to `main`. Tests must pass first.
- **Roll back:** run the workflow again on an older commit, or on the VM set `PANVISION_IMAGE` in
  `/opt/panvision/.env` to an earlier `:<sha>` tag and run `sudo docker compose up -d`.
- **Logs:** `sudo docker compose logs -f api web`.
- **Update weights:** copy the new folders into `/opt/panvision/models`, then `sudo docker compose restart api`.
