from flask import Flask, request, jsonify, render_template, url_for,session
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import redirect, secure_filename
from pdfminer.pdfdocument import PDFDocument
from pdfminer.converter import TextConverter
from scipy.spatial.distance import cosine
from pdfminer.pdfparser import PDFParser
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from datetime import datetime
from email import encoders
from flask import Response
from io import StringIO
import pandas as pd
import numpy as np
import smtplib
import warnings
import PyPDF2
import spacy
import re
import io

warnings.filterwarnings("ignore")

UPLOAD_FOLDER = ''

app = Flask(__name__)
app.secret_key = 'xyz'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

nlp = spacy.load('en_core_web_sm')
cv = CountVectorizer()

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','PDF'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def text_preprocess(sent):
    sent = re.sub('[^A-z0-9\s]','',sent)
    return sent

def get_text_from_pdfminer(path_to_text):
    output_string = StringIO()
    with open(path_to_text, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    return output_string.getvalue()

def get_text_from_pypdf(path_to_text):
    mypdf = open(path_to_text,mode='rb')
    pdfdoc = PyPDF2.PdfFileReader(mypdf)
    total_data = ""
    for i in range(pdfdoc.numPages):
        page_data = pdfdoc.getPage(i)
        total_data += page_data.extractText()

    return total_data

def str_to_datetime(datetime_str):
    try:
        datetime_object = datetime.strptime(datetime_str, '%m/%d/%Y')
    except:
        datetime_object = datetime.strptime(datetime_str, '%m/%d/%y')
    return datetime_object


def date_match(quote_data,spec_data):
    quote_dates = re.findall(r'\d+[/]\d+[/]\d+',quote_data)
    spec_dates = re.findall(r'\d+[/]\d+[/]\d+',spec_data)

    quote_date = []
    for i in quote_dates:
        try:
            quote_date.append(str_to_datetime(i))
        except:
            pass

    spec_date = []
    for i in spec_dates:
        try:
            spec_date.append(str_to_datetime(i))
        except:
            pass

    for i in quote_date:
        if i not in spec_date:
            return "Not Valid"
        else:
            return "All Dates in Quotation matched the Spec Doc."

def quotation_spec_extraction(quote_data,spec_data):
    spec_limits = re.findall(r'\d+?,?\d+,\d+\s+\W+\w+.*',spec_data)
    quotes = ''.join(re.findall(r'LIMITS OF INSURANCE.*000 SCHEDULE',quote_data))
    quote_limits = re.findall('([A-z\s&-]+\s[$]\s\d+?,?\d+,\d+)',quotes)

    # Preprocessing the text data into quotes and spec
    quote_limits = [text_preprocess(i) for i in quote_limits]
    spec_limits = [text_preprocess(i) for i in spec_limits]
    return quote_limits,spec_limits

def quotation_spec_match(quote_limits,spec_limits):
    quotes_dict = {}
    spec_dict = {}
    for quote in quote_limits:
        data_splits = re.findall(r'(.*)\s(\d+)',quote)[0]
        if data_splits != '':
            quotes_dict.update({data_splits[0]:data_splits[1]})
  
    for spec in spec_limits:
        data_splits = re.findall(r'(\d+)\s(.*)',spec)[0]
        if data_splits[1] != '':
            spec_dict.update({data_splits[1]:data_splits[0]})
  
    quotes_df = pd.DataFrame({'Description':list(quotes_dict.keys()),'Limit_in_Dollars':list(quotes_dict.values())})
    spec_df = pd.DataFrame({'Description':list(spec_dict.keys()),'Limit_in_Dollars':list(spec_dict.values())})

    quotes_df_X = quotes_df['Description'].values
    spec_df_X = spec_df['Description'].values

    quotes_df_X_cv = cv.fit_transform(quotes_df_X).toarray()
    spec_df_X_cv = cv.transform(spec_df_X).toarray()

    quotes_final_df = pd.DataFrame(quotes_df_X_cv,columns=cv.get_feature_names())
    quotes_final_df['Limit_in_Dollars'] = quotes_df['Limit_in_Dollars'].apply(pd.to_numeric)
    spec_final_df = pd.DataFrame(spec_df_X_cv,columns=cv.get_feature_names())
    spec_final_df['Limit_in_Dollars'] = spec_df['Limit_in_Dollars'].apply(pd.to_numeric)

    matched_quotes = []
    matched_spec = []
    matched_limit = []
    for row_in_quote,i in enumerate(quotes_final_df['Limit_in_Dollars']):
        for row_in_spec,j in enumerate(spec_final_df['Limit_in_Dollars']):
            if i==j:
                cos_val = cosine(quotes_final_df.iloc[row_in_quote,:-1],spec_final_df.iloc[row_in_spec,:-1])
                if cos_val < 0.55:
                    print("Matched Quotes and Specs are below")
                    print("-"*40)
                    print(quote_limits[row_in_quote],'|||||',spec_limits[row_in_spec])
                    matched_quotes.append(quote_limits[row_in_quote])
                    matched_spec.append(spec_limits[row_in_spec])
                    matched_limit.append("MATCHED")
                    print("="*40)
    
    unmatched_quotes = [i for i in quote_limits if i not in matched_quotes]
    print("\n\nUnmatched Quotes are")
    print('-'*40)
    print(unmatched_quotes)
    
    matched_df = pd.DataFrame({"Quotations":matched_quotes,
                              "Limits":matched_limit,
                              "Specifications":matched_spec})
    unmatched_df = pd.DataFrame({"Quotations":unmatched_quotes,
                                "Limits":["NOT MATCHED" for i in range(len(unmatched_quotes))]})
    df = pd.concat([matched_df,unmatched_df],axis=0)
    
    return df

def organization_match(quote,spec):
    quote_doc = nlp(quote)
    quote_orgs = []
    for ent in quote_doc.ents: 
        if ent.label_ == 'ORG':
            if ent.text not in quote_orgs:
                quote_orgs.append(ent.text)
    
    quote_org = []
    for i in quote_orgs:
        org = re.findall(r'.*[Ii]nsurance [Cc]orp',i.lower())
        if org not in quote_org and len(org)>0:
            quote_org.append(org)
            
    for i in quote_org:
        if i[0] in spec.lower():
            return i[0].upper()
        else:
            return "No Organization Found"


def send_email(to_address,body):
    fromaddr = "gavinash514@gmail.com" # Change to Your Email Address
    toaddress = [to_address]
    for toaddr in toaddress:
        # instance of MIMEMultipart
        msg = MIMEMultipart()

        # storing the senders email address
        msg['From'] = fromaddr

        # storing the receivers email address
        msg['To'] = toaddr

        # storing the subject
        msg['Subject'] = "Quotation and Spec Data Match Analysis"

        # string to store the body of the mail
        body = body

        # attach the body with the msg instance
        msg.attach(MIMEText(body, 'plain'))

        # open the file to be sent
        # file_name = filename
        # attachment = file

        # instance of MIMEBase and named as p
        # p = MIMEBase('application', 'octet-stream')

        # # To change the payload into encoded form
        # p.set_payload((attachment))

        # # encode into base64
        # encoders.encode_base64(p)

        # p.add_header('Content-Disposition', "attachment; filename= %s" % file_name)

        # # attach the instance 'p' to instance 'msg'
        # msg.attach(p)

        # creates SMTP session
        s = smtplib.SMTP('smtp.gmail.com', 587)

        # start TLS for security
        s.starttls()

        # Authentication
        s.login(fromaddr, "aVI65738") # CHANGE TO YOUR PASSWORD 

        # Converts the Multipart msg into a string
        text = msg.as_string()

        # sending the mail
        try:
            s.sendmail(fromaddr, toaddr, text)

        except:
            s.sendmail(fromaddr, toaddr,"File Size Exceeded could not send a file")

        # terminating the session
    s.quit()


@app.route('/', methods=['GET', 'POST'])
def texmining():
    if request.method == 'POST':
        ext_quote = request.files['quotefile']
        print("Quote1",ext_quote.filename)
        ext_spec = request.files['specfile']
        to_email = request.form['email']
        sfname1 = ''+str(secure_filename('quote1.PDF'))
        sfname2 = ''+str(secure_filename('spec1.PDF'))
        ext_quote.save(sfname1)
        ext_spec.save(sfname2)
        filename_quote = ext_quote.filename
        filename_spec = ext_spec.filename
        quote1 = get_text_from_pypdf('./quote1.PDF')
        spec1 = get_text_from_pdfminer('./spec1.PDF')
        # print("Quote File Name - ", quote1)
        # print("Spec File Name - ", spec1)

        # Date Match
        date_output = date_match(quote_data=quote1,spec_data=spec1)

        # # Quotations Match
        quote_limits,spec_limits = quotation_spec_extraction(quote_data=quote1,spec_data=spec1)
        qsdf = quotation_spec_match(quote_limits=quote_limits,spec_limits=spec_limits)
        # s = io.StringIO()
        # dataset.to_csv(s,index=False)
        # my_csv = s.getvalue()

        # # Organization Match
        organizations = organization_match(quote1,spec1)
        # return render_template('home.html')

        send_email(to_address = to_email,body="Quotations Match - \n-------------------\n{}\n\nDates Match - {}\n\nOrganizations Match - {}".format(qsdf,date_output,organizations))
        return render_template('home.html',tables=[qsdf.to_html(classes='data')], titles=qsdf.columns.values, 
                                date_outputs = "Dates Match - \n"+date_output, organization = "Organizations - \n"+organizations)
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)