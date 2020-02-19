# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:10:07 2020

@author: ad1006362
"""

import requests
#from requests.auth import HTTPBasicAuth
#from getpass import getpass

import urllib

import xml.etree.ElementTree as ET

import pandas as pd 

from pathlib import Path

print(urllib.request.getproxies())

# Replace with the correct URL
#projects_url = "https://mea.aconex.com/api/projects/"
projects_url = "https://apidev.aconex.com/api/projects/"
sanbox_header={'X-Application-Key': '0c3d68fa-4348-4eee-8a1b-eff1c7b2f030'}

# By using a context manager, you can ensure the resources used by
# the session will be released after use
with requests.Session() as session:
    session.auth = ('poleary', "Auth3nt1c")

    # Instead of requests.get(), you'll use session.get()
    projectresponse = session.get(projects_url,
                           headers=sanbox_header,                          
                           verify=False
                          )

# You can inspect the response just like you did before
print(projectresponse.headers)
projectresponse.encoding ='utf-8'
#print(projectresponse.text)
projectroot = ET.fromstring(projectresponse.content)

projectid_list=[]
for child in projectroot.iter('ProjectId'):
    projectid_list.append(child.text)


for i in range(1,3):
    print(projectid_list[i])

#sampleprojectid = projectid_list[0]
sampleprojectid=1879048428
# https://apidev.aconex.com/api/projects/{PROJECTID}/register/schema
docschema_url = " https://apidev.aconex.com/api/projects/{}/register/schema"
docschema_url = docschema_url.format(sampleprojectid)
with requests.Session() as session:
    session.auth = ('poleary', "Auth3nt1c")
    
    # Instead of requests.get(), you'll use session.get()
    schemaresponse = session.get(docschema_url,
                           headers=sanbox_header,                          
                           verify=False
                          )


docschema = schemaresponse.text
print(docschema)

sampleprojectid=1879048428
# https://apidev.aconex.com/api/projects/{PROJECTID}/register?return_fields=trackingid
doclist_url =  "https://apidev.aconex.com/api/projects/{}/register?return_fields=trackingid"
doclist_url = doclist_url.format(sampleprojectid)

with requests.Session() as session:
    session.auth = ('poleary', "Auth3nt1c")
    
    # Instead of requests.get(), you'll use session.get()
    doclistresponse = session.get(doclist_url,
                           headers=sanbox_header,                          
                           verify=False
                          )


doclisttext = doclistresponse.text
print(doclisttext)


# get all versions based on tracking number
sampleTrackId=271341877549075742
#https://apidev.aconex.com/api/projects/{PROJECTID}/register?search_query=trackingid:{TRACKINGID}&return_fields=current,trackingid,revision,versionnumber,registered&show_document_history=true
docversions_url="https://apidev.aconex.com/api/projects/{}/register?search_query=trackingid:{}&return_fields=current,trackingid,revision,versionnumber,registered&show_document_history=true"
docversions_url=docversions_url.format(sampleprojectid,sampleTrackId)
with requests.Session() as session:
    session.auth = ('poleary', "Auth3nt1c")
    
    # Instead of requests.get(), you'll use session.get()
    docversionsresponse = session.get(docversions_url,
                           headers=sanbox_header,                          
                           verify=False
                          )


docversionstext = docversionsresponse.text
print(docversionstext)


# search documents
# https://apidev.aconex.com/api/projects/{PROJECTID}/register?search_query=doctype:"Shop Drawing" AND Door&return_fields=approved,asBuiltRequired,attribute1,attribute2,attribute3,attribute4,author,authorisedBy,category&search_type=NUMBER_LIMITED&search_result_size=75&sort_field=docno&sort_direction=ASC&show_document_history=true
search_string="pdf"
doc_search_url=" https://apidev.aconex.com/api/projects/{}/register?search_query=filetype:\"{}\" AND Door&return_fields=approved,asBuiltRequired,attribute1,attribute2,attribute3,attribute4,author,authorisedBy,category&search_type=NUMBER_LIMITED&search_result_size=75&sort_field=docno&sort_direction=ASC&show_document_history=true"
doc_search_url=doc_search_url.format(sampleprojectid, search_string)

with requests.Session() as session:
    session.auth = ('poleary', "Auth3nt1c")
    
    # Instead of requests.get(), you'll use session.get()
    searchresponse = session.get(doc_search_url,
                           headers=sanbox_header,                          
                           verify=False
                          )

searchresults = searchresponse.text
print(searchresults)

# get the meta data based on the document number
sampleDocId=271341877549084788
# https://apidev.aconex.com/api/projects/{PROJECTID}/register/{DOCUMENTID}/metadata
docmetadata_url="https://apidev.aconex.com/api/projects/{}/register/{}/metadata"
docmetadata_url=docmetadata_url.format(sampleprojectid,sampleDocId)
with requests.Session() as session:
    session.auth = ('poleary', "Auth3nt1c")
    
    # Instead of requests.get(), you'll use session.get()
    docmetadataresponse = session.get(docmetadata_url,
                           headers=sanbox_header,                          
                           verify=False
                          )


docmetadatatext = docmetadataresponse.text
print(docmetadatatext)

# get the file content based on the document number
sampleDocId=271341877549084788
# https://apidev.aconex.com/api/projects/{PROJECTID}/register/{DOCUMENTID}/markedup
doccontent_url="https://apidev.aconex.com/api/projects/{}/register/{}/markedup"
doccontent_url=doccontent_url.format(sampleprojectid,sampleDocId)
with requests.Session() as session:
    session.auth = ('poleary', "Auth3nt1c")
    
    # Instead of requests.get(), you'll use session.get()
    doccontentresponse = session.get(doccontent_url,
                           headers=sanbox_header,                          
                           verify=False
                          )
    print(doccontentresponse.headers.get('content-type'))


doccontent = doccontentresponse.content
filename = Path('d:\\Temp\\Aconex-{}.pdf'.format(sampleDocId))
filename.write_bytes(doccontent)

# Audit log (integrity check)
#https://apidev.aconex.com/api/projects/{PROJECTID}/register/integrity?everythingsince=<2001-10-26T19:00:00.000Z>&show_document_history=true
sincedate ="2001-10-26T19:00:00.000Z"
audit_doc_url=" https://apidev.aconex.com/api/projects/{}/register/integrity?everythingsince={}&show_document_history=true"
audit_doc_url=audit_doc_url.format(sampleprojectid, sincedate)
with requests.Session() as session:
    session.auth = ('poleary', "Auth3nt1c")
    
    # Instead of requests.get(), you'll use session.get()
    integrityresponse = session.get(audit_doc_url,
                           headers=sanbox_header,                          
                           verify=False
                          )
integritytext= integrityresponse.text
print(integritytext)