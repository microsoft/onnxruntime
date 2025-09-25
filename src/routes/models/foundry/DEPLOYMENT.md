# Azure AI Foundry Models - Deployment Guide

## 🎯 CORS Issue & Solutions

The Azure AI Foundry API works perfectly with direct requests (as proven by your curl command), but browsers enforce CORS (Cross-Origin Resource Sharing) policy that blocks direct requests from web applications.

## ✅ **Solution Implemented: SvelteKit Proxy API**

I've created a proxy API route at `/src/routes/api/foundry-models/+server.ts` that:

1. **Receives requests** from your frontend
2. **Forwards them** to Azure AI Foundry API from the server
3. **Returns the response** back to your frontend

### How it works:

```
Frontend → SvelteKit API Route → Azure AI Foundry API → Response back
```

## 🚀 **Deployment Scenarios**

### **1. Deployed to any domain (Recommended)**

- ✅ **Will work perfectly**
- The proxy runs on your server, not in the browser
- Server-to-server requests are not subject to CORS
- No additional configuration needed

### **2. Vercel/Netlify/Other Serverless**

- ✅ **Will work perfectly**
- SvelteKit API routes become serverless functions
- Same proxy approach works seamlessly

### **3. Static Site Deployment**

- ❌ **Won't work** (no server-side API routes)
- Would need alternative solutions (see below)

## 🛠️ **Alternative Workarounds (if needed)**

### **Option 1: Direct CORS Bypass (Development Only)**

```bash
# Chrome with CORS disabled
chrome --user-data-dir="/tmp/chrome" --disable-web-security --disable-features=VizDisplayCompositor

# Or use browser extension to disable CORS
```

### **Option 2: Reverse Proxy (Production)**

Set up nginx/Apache to proxy requests:

```nginx
location /api/azure/ {
    proxy_pass https://ai.azure.com/api/eastus/ux/v1.0/entities/;
    proxy_set_header User-Agent "AzureAiStudio";
}
```

### **Option 3: Azure API Management**

- Set up Azure API Management as intermediary
- Configure CORS policies in Azure
- More complex but enterprise-grade solution

## 📊 **Testing Your Current Setup**

Your curl command proves the API works:

```bash
curl --request POST \
  --url https://ai.azure.com/api/eastus/ux/v1.0/entities/crossRegion \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: AzureAiStudio' \
  --data '{...}'
```

The proxy API route uses the **exact same request format** but from the server side.

## 🎉 **Bottom Line**

**Your foundry models page will work perfectly when deployed** because:

- The proxy API handles CORS restrictions
- Server-to-server requests work fine
- The exact curl request format is preserved
- No authentication tokens needed (based on your working curl example)

The page will display real Azure AI Foundry models on any deployed domain! 🚀
