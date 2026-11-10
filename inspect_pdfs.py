from pypdf import PdfReader
import pdfplumber, os
for path in [r'C:\Users\Admin\Downloads\mistake pdf.pdf',r'C:\Users\Admin\Downloads\Content.pdf']:
    print('\nFILE',path)
    reader=PdfReader(path)
    print('PAGES',len(reader.pages))
    with pdfplumber.open(path) as pdf:
        for i,p in enumerate(pdf.pages):
            text=p.extract_text() or ''
            print(f'---PAGE {i+1} ({len(text)} chars)---')
            print(text)
            ann=reader.pages[i].get('/Annots') or []
            for a in ann:
                o=a.get_object(); sub=o.get('/Subtype')
                if sub=='/Highlight':
                    print('HIGHLIGHT',o.get('/C'),o.get('/QuadPoints'),o.get('/Contents'),o.get('/T'))
