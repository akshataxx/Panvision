#!/usr/bin/env bash
# One-time Google Cloud setup for PanVision. Paste into Google Cloud Shell (gcloud is preinstalled there).
# Creates: Artifact Registry repo, VM + static IP + firewall, GitHub keyless deploy identity.
# Prints the values to enter as GitHub repository variables at the end.
set -euo pipefail

PROJECT=$(gcloud config get-value project 2>/dev/null)
REGION=asia-southeast1
ZONE=asia-southeast1-a
VM=panvision
REPO=hrihaanG/Panvision
MACHINE=e2-standard-2

[ -n "$PROJECT" ] || { echo "Select a project first: gcloud config set project YOUR_PROJECT_ID"; exit 1; }
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
VM_SA=panvision-vm@$PROJECT.iam.gserviceaccount.com
DEPLOY_SA=github-deployer@$PROJECT.iam.gserviceaccount.com
echo "Setting up PanVision in project $PROJECT ($REGION)"

gcloud services enable compute.googleapis.com artifactregistry.googleapis.com \
  iamcredentials.googleapis.com iap.googleapis.com

# Image registry
gcloud artifacts repositories create panvision --repository-format=docker --location=$REGION

# VM identity: may only pull images
gcloud iam service-accounts create panvision-vm --display-name="PanVision VM"
gcloud artifacts repositories add-iam-policy-binding panvision --location=$REGION \
  --member=serviceAccount:$VM_SA --role=roles/artifactregistry.reader

# VM, fixed IP, firewall (web open; SSH only via Google's IAP range, which browser SSH also uses)
gcloud compute addresses create panvision-ip --region=$REGION
gcloud compute instances create $VM --zone=$ZONE --machine-type=$MACHINE \
  --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud --boot-disk-size=50GB \
  --service-account=$VM_SA --scopes=cloud-platform --address=panvision-ip \
  --metadata=enable-oslogin=TRUE --tags=panvision
gcloud compute firewall-rules create panvision-web --allow=tcp:80,tcp:443 --target-tags=panvision
gcloud compute firewall-rules create panvision-iap-ssh --allow=tcp:22 \
  --source-ranges=35.235.240.0/20 --target-tags=panvision

# GitHub Actions identity: push images, SSH through IAP with sudo
gcloud iam service-accounts create github-deployer --display-name="GitHub deployer"
gcloud artifacts repositories add-iam-policy-binding panvision --location=$REGION \
  --member=serviceAccount:$DEPLOY_SA --role=roles/artifactregistry.writer
for role in roles/compute.viewer roles/compute.osAdminLogin roles/iap.tunnelResourceAccessor; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member=serviceAccount:$DEPLOY_SA \
    --role=$role --condition=None > /dev/null
done
gcloud iam service-accounts add-iam-policy-binding $VM_SA \
  --member=serviceAccount:$DEPLOY_SA --role=roles/iam.serviceAccountUser

# Keyless trust: only workflows from $REPO can act as github-deployer
gcloud iam workload-identity-pools create github --location=global --display-name="GitHub"
gcloud iam workload-identity-pools providers create-oidc github --location=global \
  --workload-identity-pool=github --issuer-uri=https://token.actions.githubusercontent.com \
  --attribute-mapping=google.subject=assertion.sub,attribute.repository=assertion.repository \
  --attribute-condition="assertion.repository=='$REPO'"
gcloud iam service-accounts add-iam-policy-binding $DEPLOY_SA --role=roles/iam.workloadIdentityUser \
  --member=principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/$REPO

# Docker on the VM, able to pull from the registry; folder for compose files, secrets and weights
echo "Waiting for the VM to boot..."; sleep 45
gcloud compute ssh $VM --zone=$ZONE --tunnel-through-iap --quiet --command "
  set -e
  curl -fsSL https://get.docker.com | sudo sh
  sudo gcloud auth configure-docker $REGION-docker.pkg.dev --quiet
  sudo install -d /opt/panvision/models"

IP=$(gcloud compute addresses describe panvision-ip --region=$REGION --format='value(address)')
cat <<EOF

Done. VM IP: $IP
Add these on GitHub → $REPO → Settings → Secrets and variables → Actions → Variables:

  GCP_PROJECT_ID    $PROJECT
  GCP_REGION        $REGION
  GCP_ZONE          $ZONE
  GCP_VM_NAME       $VM
  GCP_DEPLOY_SA     $DEPLOY_SA
  GCP_WIF_PROVIDER  projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github
  PANVISION_DOMAIN  ${IP//./-}.sslip.io   (or your own domain pointed at $IP)
EOF
