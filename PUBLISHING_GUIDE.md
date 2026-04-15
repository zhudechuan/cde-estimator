# Publishing cde-estimator — Step-by-Step

Follow these steps in order. The whole process takes about 15 minutes.

---

## Step 1: Create the GitHub Repository

1. Go to https://github.com/new
2. Repository name: `cde-estimator`
3. Description: `Constrained Dantzig-type Estimator for high-dimensional portfolio selection`
4. Set to **Public**
5. Do NOT initialize with README/.gitignore/license (we already have them)
6. Click **Create repository**

## Step 2: Push Your Code

Open a terminal in the `cde-estimator` folder and run:

```bash
cd /path/to/cde-estimator

git init
git add .
git commit -m "Initial release v0.1.0"
git branch -M main
git remote add origin https://github.com/zhudechuan/cde-estimator.git
git push -u origin main
```

## Step 3: Create PyPI & TestPyPI Accounts

### PyPI (the real package index)
1. Go to https://pypi.org/account/register/
2. Register with your email
3. Enable 2FA (required for publishing)

### TestPyPI (for dry-run testing)
1. Go to https://test.pypi.org/account/register/
2. Register (separate account from PyPI)

## Step 4: Set Up Trusted Publishing (Recommended)

This lets GitHub Actions publish to PyPI without storing API tokens as secrets. It's the most secure approach.

### On PyPI:
1. Log in at https://pypi.org
2. Go to **Account Settings** → **Publishing** → **Add a new pending publisher**
3. Fill in:
   - PyPI project name: `cde-estimator`
   - Owner: `zhudechuan`
   - Repository name: `cde-estimator`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
4. Click **Add**

### On TestPyPI:
1. Log in at https://test.pypi.org
2. Same steps as above, but set environment name to `testpypi`

### On GitHub:
1. Go to your repo → **Settings** → **Environments**
2. Create two environments: `pypi` and `testpypi`
3. (Optional) Add a protection rule on `pypi` requiring manual approval — this gives you a chance to verify the TestPyPI upload before the real one goes out

## Step 5: First Publish — Create a GitHub Release

1. Go to your repo on GitHub
2. Click **Releases** → **Create a new release**
3. Tag: `v0.1.0` (create new tag)
4. Title: `v0.1.0 — Initial Release`
5. Description — paste from CHANGELOG.md
6. Click **Publish release**

This triggers the `publish.yml` workflow, which will:
1. Build the package
2. Upload to TestPyPI first
3. Then upload to PyPI

You can monitor progress in the **Actions** tab.

## Step 6: Verify

After the workflow completes:

```bash
# Check it's on PyPI
pip install cde-estimator

# Or test from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ cde-estimator
```

---

## For Future Releases

1. Update `version` in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit and push
4. Create a new GitHub Release with the matching tag (e.g., `v0.2.0`)
5. The CI/CD pipeline handles the rest

---

## Alternative: Manual Publish (No CI)

If you ever need to publish manually:

```bash
pip install build twine
python -m build
twine upload dist/*           # uploads to PyPI
# or: twine upload --repository testpypi dist/*   # uploads to TestPyPI
```

You'll need an API token from https://pypi.org/manage/account/token/ for this approach.
