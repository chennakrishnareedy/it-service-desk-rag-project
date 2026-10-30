"""
Generate realistic sample IT Service Desk documents in multiple formats
(PDF, DOCX, HTML, TXT) so the RAG assistant has an approved knowledge base
to ingest.

Run:
    python src/generate_sample_docs.py
"""
from __future__ import annotations

import os
from pathlib import Path

from docx import Document
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


# ----------------------------------------------------------------------------
# Document content (approved IT knowledge base)
# ----------------------------------------------------------------------------
PASSWORD_RESET = """Corporate Password Reset Guide

Overview
This guide explains how employees can reset their corporate password using the
Self-Service Password Reset (SSPR) portal.

Steps to reset your corporate password
1. Open a browser and go to https://passwordreset.corp.local
2. Enter your corporate email address and click Next.
3. Verify your identity using the registered authentication method
   (Microsoft Authenticator push, SMS code, or security questions).
4. Enter a new password that meets the password policy.
5. Click Submit. You will see a confirmation message within 30 seconds.

Password policy requirements
- Minimum 12 characters.
- At least one uppercase letter, one lowercase letter, one number and one symbol.
- Cannot reuse any of your last 10 passwords.
- Password expires every 90 days.

Common issues
- If you do not have a registered authentication method, contact Level 1 support.
- If the portal shows "Account locked", wait 15 minutes and try again.
- After 3 failed SSPR attempts the account is locked and must be escalated to
  Level 2 support for manual unlock.
"""

VPN_GUIDE = """Corporate VPN Troubleshooting Guide

Overview
This document describes how to connect to the corporate VPN using the GlobalConnect
client and how to troubleshoot common connection problems.

Connecting to the VPN
1. Launch the GlobalConnect VPN client.
2. Select the gateway "corp-vpn-gateway".
3. Enter your corporate credentials and approve the MFA prompt.
4. Wait for the status to show "Connected".

Troubleshooting a VPN that will not connect
1. Confirm you have an active internet connection.
2. Make sure the GlobalConnect client is updated to version 7.2 or later.
3. Restart the GlobalConnect client and try again.
4. Flush the DNS cache and reconnect.
5. If you are on a public Wi-Fi captive portal, open a browser and log in first.
6. If it still fails, restart your laptop and retry.

VPN error codes
- VPN-401: Authentication failed. Re-enter your credentials and approve MFA.
- VPN-403: Access denied. Your account is not authorized for this VPN gateway.
  You must request VPN access through the IT access portal and wait for approval.
- VPN-408: Connection timed out. Check your network and retry.
- VPN-500: Gateway server error. Escalate to Level 2 support if it persists for
  more than 30 minutes.

Escalation
If the VPN still fails after completing all troubleshooting steps, or if you see
error VPN-500 for more than 30 minutes, escalate the issue to Level 2 support.
"""

SOFTWARE_INSTALL = """Approved Software Installation Guide

Overview
This guide explains how to install approved software, including the approved
Python version, using the Company Software Center.

Installing approved software
1. Open the Company Software Center from the Start menu.
2. Search for the application you need.
3. Click Install. Approved applications install without admin rights.
4. If the application is not listed, submit a software request ticket.

Installing the approved Python version
The approved corporate Python version is Python 3.10.
1. Open the Company Software Center.
2. Search for "Python 3.10 (Approved)".
3. Click Install and wait for completion.
4. Open a terminal and run "python --version" to confirm 3.10 is installed.
Do not download Python from the public internet; only the Software Center build
is approved and security-scanned.

Common issues
- "Installation failed" usually means a pending Windows update. Reboot and retry.
- If you need a version other than the approved one, this requires a security
  exception and must be escalated to Level 2 support.
"""

NETWORK_FAQ = """IT Service Desk Frequently Asked Questions

Q: How can I reset my corporate password?
A: Use the Self-Service Password Reset portal at https://passwordreset.corp.local
   and follow the identity verification steps. See the Password Reset Guide.

Q: What should I do if the VPN is not connecting?
A: Verify your internet, update the GlobalConnect client, restart it, flush DNS,
   and reconnect. See the VPN Troubleshooting Guide for error-code meanings.

Q: How do I install the approved Python version?
A: Install "Python 3.10 (Approved)" from the Company Software Center.

Q: My network is slow. What can I do?
A: Restart your router or dock, connect via Ethernet if possible, and run the
   Network Diagnostics tool from the Software Center. If packet loss is above 5%,
   open a network ticket.

Q: What does error code VPN-403 mean?
A: VPN-403 means access denied: your account is not authorized for that VPN
   gateway. Request VPN access through the IT access portal.

Q: When should an issue be escalated to Level 2 support?
A: Escalate to Level 2 when: an account must be manually unlocked, a VPN-500
   error persists over 30 minutes, a non-approved software version is required,
   or when Level 1 troubleshooting steps do not resolve the issue.
"""

TICKETS = """Previously Resolved Support Tickets

Ticket #10234 - Category: Password
Issue: User could not log in after password expired.
Resolution: Guided user through the Self-Service Password Reset portal. User
reset the password successfully. Root cause: 90-day password expiry.

Ticket #10251 - Category: VPN
Issue: User reported VPN error VPN-403 when connecting.
Resolution: User was not a member of the VPN access group. Submitted an access
request; after approval the user connected successfully.

Ticket #10288 - Category: Software
Issue: User needed Python but installed an unapproved version from the internet.
Resolution: Removed the unapproved build and installed "Python 3.10 (Approved)"
from the Software Center. Reminded user of the approved-software policy.

Ticket #10299 - Category: Network
Issue: Intermittent network drops on Wi-Fi.
Resolution: Updated the wireless driver and switched the user to the 5 GHz SSID.
Packet loss dropped below 1%. Issue resolved at Level 1.

Ticket #10312 - Category: VPN
Issue: VPN-500 gateway error persisting for over an hour.
Resolution: Escalated to Level 2. Gateway node was restarted by the network team.
"""


# ----------------------------------------------------------------------------
# Writers for each format
# ----------------------------------------------------------------------------
def write_txt(name: str, text: str) -> None:
    (DOCS_DIR / name).write_text(text, encoding="utf-8")


def write_pdf(name: str, text: str) -> None:
    doc = SimpleDocTemplate(str(DOCS_DIR / name), pagesize=LETTER)
    styles = getSampleStyleSheet()
    flow = []
    for line in text.strip().split("\n"):
        if line.strip() == "":
            flow.append(Spacer(1, 8))
        else:
            flow.append(Paragraph(line.replace("&", "&amp;"), styles["Normal"]))
    doc.build(flow)


def write_docx(name: str, text: str) -> None:
    doc = Document()
    lines = text.strip().split("\n")
    if lines:
        doc.add_heading(lines[0], level=1)
    for line in lines[1:]:
        doc.add_paragraph(line)
    doc.save(str(DOCS_DIR / name))


def write_html(name: str, text: str) -> None:
    lines = text.strip().split("\n")
    title = lines[0] if lines else "IT Document"
    body = "".join(
        f"<p>{line}</p>\n" if line.strip() else "<br/>\n" for line in lines[1:]
    )
    html = (
        f"<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body>\n<h1>{title}</h1>\n{body}</body></html>"
    )
    (DOCS_DIR / name).write_text(html, encoding="utf-8")


def main() -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)
    # Mix of formats to exercise the multi-format ingestion pipeline.
    write_pdf("password_reset_guide.pdf", PASSWORD_RESET)
    write_pdf("vpn_troubleshooting_guide.pdf", VPN_GUIDE)
    write_docx("software_installation_guide.docx", SOFTWARE_INSTALL)
    write_html("it_faq.html", NETWORK_FAQ)
    write_txt("resolved_tickets.txt", TICKETS)
    print(f"Sample documents written to: {DOCS_DIR}")
    for p in sorted(DOCS_DIR.iterdir()):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
