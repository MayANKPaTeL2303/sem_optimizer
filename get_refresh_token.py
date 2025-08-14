from google_auth_oauthlib.flow import InstalledAppFlow

# Replace with your own OAuth client ID & secret
CLIENT_ID = "59953708455-fp3o6uilf6srkm6mnd1gof0mkoserqvd.apps.googleusercontent.com"
CLIENT_SECRET = "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"

# Scope for Google Ads API
SCOPES = ["https://www.googleapis.com/auth/adwords"]

flow = InstalledAppFlow.from_client_secrets_file(
    'client_secret_59953708455-fp3o6uilf6srkm6mnd1gof0mkoserqvd.apps.googleusercontent.com.json',
    scopes=['https://www.googleapis.com/auth/adwords']
)

credentials = flow.run_local_server(port=8081)
print("Refresh token:", credentials.refresh_token)
