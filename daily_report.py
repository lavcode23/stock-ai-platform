from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

def generate_daily_pdf(metrics, closed_trades, path="daily_report.pdf"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=A4)
    story = []

    story.append(Paragraph(f"<b>Daily Trading Report</b>", styles["Title"]))
    story.append(Paragraph(datetime.now().strftime("%Y-%m-%d"), styles["Normal"]))
    story.append(Spacer(1, 10))

    for k,v in metrics.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    story.append(Spacer(1, 10))

    if closed_trades is not None:
        for _,r in closed_trades.tail(10).iterrows():
            story.append(Paragraph(
                f"{r['symbol']} | PnL: {r['pnl']}",
                styles["Normal"]
            ))

    doc.build(story)
    return path
