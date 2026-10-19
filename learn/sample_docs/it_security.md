# Halcyon Robotics — IT Security Policy

This policy applies to every employee, contractor and intern. Questions go to the security team at security@halcyon.example.

## Passwords

Passwords must be at least 14 characters long and must not be reused across services. All passwords are stored in the company password manager at vault.halcyon.internal; storing passwords in browsers, notes or spreadsheets is not allowed. Shared team credentials live in shared vault collections, never in chat.

## Multi-factor authentication

Multi-factor authentication is required on every company account. Hardware security keys are mandatory for anyone with production access; authenticator apps are acceptable for everyone else. SMS codes are not accepted as a second factor. Each engineer with production access is issued two hardware keys: one for daily use and one kept in a safe place as a backup.

## VPN

Internal services are only reachable through the WireGuard VPN. Download your personal configuration from the IT portal; configurations are tied to a single device and must not be copied. VPN sessions disconnect automatically after 12 hours and must be re-authenticated.

## Devices

All laptops use full-disk encryption and lock automatically after 5 minutes of inactivity. A lost or stolen device must be reported to security@halcyon.example within 1 hour of noticing it is missing, so the device can be wiped remotely. Personal devices may access email and chat through the mobile management app, but never source code.

## Patching

Operating system and browser updates are installed automatically. Critical security patches must be applied within 72 hours of release; the IT team will force a restart if a laptop falls behind.

## Phishing

Report suspicious emails with the "Report phishing" button in the mail client. Never forward a suspected phishing email to colleagues. The security team runs a simulated phishing exercise every quarter; clicking a simulated link triggers a short training module, not a penalty.

## Data classification

Data is classified as Public, Internal, Confidential or Restricted. Customer data and robot telemetry are Confidential. Source code for the navigation stack is Restricted and may only be stored in the company Git server.
