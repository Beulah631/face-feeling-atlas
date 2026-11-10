from pypdf import PdfReader
import pdfplumber
path=r'C:\Users\Admin\Downloads\mistake pdf.pdf'
r=PdfReader(path)
with pdfplumber.open(path) as pdf:
 for i,page in enumerate(r.pages):
  for a in page.get('/Annots') or []:
   o=a.get_object()
   if o.get('/Subtype')!='/Highlight': continue
   qs=o.get('/QuadPoints') or []
   parts=[]
   pp=pdf.pages[i]
   for j in range(0,len(qs),8):
    x0=min(qs[j],qs[j+2],qs[j+4],qs[j+6]); x1=max(qs[j],qs[j+2],qs[j+4],qs[j+6]);
    y0=min(qs[j+1],qs[j+3],qs[j+5],qs[j+7]); y1=max(qs[j+1],qs[j+3],qs[j+5],qs[j+7]);
    rect=(x0,pp.height-y1,x1,pp.height-y0)
    words=pp.extract_words(x_tolerance=2,y_tolerance=3)
    sel=[w['text'] for w in words if w['x1']>=rect[0] and w['x0']<=rect[2] and w['bottom']>=rect[1]-2 and w['top']<=rect[3]+2]
    parts.append(' '.join(sel))
   print(f'PAGE {i+1}: ' + ' | '.join(parts))
