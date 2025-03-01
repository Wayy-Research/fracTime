# FracTime Deployment Guide

## Local Deployment

Running FracTime locally is straightforward:

```bash
# Option 1: Use the start script (recommended)
./start_app.sh

# Option 2: Run directly with Streamlit
streamlit run Home.py
```

## Streamlit Cloud Deployment

### Step 1: GitHub Repository Setup

1. Ensure your repository is pushed to GitHub under the `rcgalbo` account
2. Make sure the repository is either public or properly shared with Streamlit Cloud

### Step 2: Streamlit Cloud Setup

1. Go to [Streamlit Cloud](https://share.streamlit.io/)
2. Sign out of any existing accounts
3. Sign in with your GitHub account (rcgalbo)
4. Create a new app with these settings:
   - Repository: `rcgalbo/fracTime`
   - Branch: `main`
   - Main file path: `Home.py`
   - Advanced settings: Check "Enable CORS for connection to API"

### Step 3: Troubleshooting Access Issues

If you encounter the error "You do not have access to this app or it does not exist":

1. **Check Repository Visibility**:
   - Ensure your repository is public or properly shared

2. **Check GitHub Connection**:
   - Go to your Streamlit account settings
   - Verify that GitHub is connected with the correct permissions
   - Re-authorize if necessary

3. **Create a New Deployment**:
   - Delete the existing app from Streamlit Cloud
   - Create a new app with the same settings
   - This often resolves permission caching issues

4. **Verify Owner Settings**:
   - If deploying to an organization, ensure your GitHub account has admin access
   - Check that both `rcgalbo@gmail.com` and `wayyresearch@gmail.com` are added as collaborators

5. **Contact Streamlit Support**:
   - If issues persist, contact support with screenshots and repository details
   - They can manually check permission settings that might not be visible to you

### Step 4: Post-Deployment Verification

After successful deployment:

1. Test all pages and features
2. Verify data loading and visualization functionality
3. Check that page navigation works correctly

## Configuration Files

The `.streamlit` directory contains important configuration files:

- `config.toml`: Core Streamlit settings
- `deploy.toml`: Deployment configuration
- `secrets.toml`: Authentication secrets (not pushed to GitHub)

Ensure these files are properly configured before deployment.