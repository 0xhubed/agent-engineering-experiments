"""Synthetic termsheet PDFs for live checks. Entirely fictional; needs reportlab (dev dependency)."""
from __future__ import annotations

from pathlib import Path


def make_termsheet_pdf(path: str | Path, *, product: str = "Example note", barrier: str = "60%",
                       shares: str = "4.0") -> None:
    """Twelve pages: six sections, each a content page plus a filler continuation page."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    sections = [("1 Product Terms", [f"Product: {product} (synthetic)", "Issuer: Example Co. (synthetic)",
                                     "Underlying: Example Index", "Denomination: CHF 5,000"]),
                ("2 Barrier", [f"The barrier level is {barrier} of the initial fixing level.",
                               "Barrier observation is continuous."]),
                ("3 Coupon", ["Coupon: 4.5% p.a., paid semi-annually.",
                              "Coupon payment dates: 15 June 2026 and 15 December 2026."]),
                ("4 Redemption", ["If no barrier event occurs, redemption at 100% of denomination.",
                                  f"Otherwise physical delivery of {shares} shares per note."]),
                ("5 Risk Factors", ["Investors may lose their entire investment.", "Liquidity may be limited."]),
                ("6 Selling Restrictions", ["The product may not be offered in the United States."])]
    c = canvas.Canvas(str(path), pagesize=A4)
    for title, lines in sections:
        for part in (1, 2):
            c.setFont("Helvetica-Bold", 16)
            c.drawString(72, 770, title if part == 1 else f"{title} (continued)")
            c.setFont("Helvetica", 11)
            for i, line in enumerate(lines if part == 1 else ["General provisions apply. " * 3]):
                c.drawString(72, 740 - 18 * i, line)
            c.showPage()
    c.save()
