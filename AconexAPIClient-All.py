# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:33:33 2020

@author: ad1006362
"""
'''
Class wich connect to the Aconex Cloud solution to geth the Project worspace items and synchronizes with the DMS system

'''

import requests

import xml.etree.ElementTree as ET

import os

import re 

from pathlib import Path

import xml.dom.minidom


class AconexAPIClient:
    
    def __init__(self, projectsurl, secreatkey, username, password):
        self.projectsurl = projectsurl
        self.headers = {'X-Application-Key': secreatkey}
        self.username = username
        self.password = password
    
    def loadProjects(self):
        
        return self.getResponse(self.projectsurl, self.headers, self.username, self.password)
    
    
    def loadDocumentRegisterSchema(self, projectId):
        
        # https://apidev.aconex.com/api/projects/{PROJECTID}/register/schema
        docschema_url = " https://apidev.aconex.com/api/projects/{}/register/schema"
        docschema_url = docschema_url.format(projectId)
        
        return self.getResponse(docschema_url, self.headers, self.username, self.password)
    
    
    def loadDocumentRegister(self, projectId):
        
        # https://apidev.aconex.com/api/projects/{PROJECTID}/register?return_fields=trackingid
        doclist_url =  "https://apidev.aconex.com/api/projects/{}/register?return_fields=trackingid"
        doclist_url = doclist_url.format(projectId)
        
        return self.getResponse(doclist_url,self.headers, self.username, self.password)
    
    
    def loadDocumentVersions(self, projectId, trackingId):
        
        #https://apidev.aconex.com/api/projects/{PROJECTID}/register?search_query=trackingid:{TRACKINGID}&return_fields=current,trackingid,revision,versionnumber,registered&show_document_history=true
        docversions_url="https://apidev.aconex.com/api/projects/{}/register?search_query=trackingid:{}&return_fields=current,trackingid,revision,versionnumber,registered&show_document_history=true"
        docversions_url=docversions_url.format(projectId,trackingId)
        
        return self.getResponse(docversions_url, self.headers, self.username, self.password)
    
    def loadDocumentMetadata(self, projectId, documentId):
        
        # https://apidev.aconex.com/api/projects/{PROJECTID}/register/{DOCUMENTID}/metadata
        docmetadata_url="https://apidev.aconex.com/api/projects/{}/register/{}/metadata"
        docmetadata_url=docmetadata_url.format(projectId,documentId)
        
        return self.getResponse(docmetadata_url, self.headers, self.username, self.password)
    
    def downloadContentFile(self, projectId, documentId):
     
        # https://apidev.aconex.com/api/projects/{PROJECTID}/register/{DOCUMENTID}/markedup
        doccontent_url="https://apidev.aconex.com/api/projects/{}/register/{}/markedup"
        doccontent_url=doccontent_url.format(projectId,documentId)
        
        return self.getResponse(doccontent_url, self.headers, self.username, self.password)
    
    def searchDocuments(self, projectId, searchString="pdf"):
        
        # https://apidev.aconex.com/api/projects/{PROJECTID}/register?search_query=doctype:"Shop Drawing" AND Door&return_fields=approved,asBuiltRequired,attribute1,attribute2,attribute3,attribute4,author,authorisedBy,category&search_type=NUMBER_LIMITED&search_result_size=75&sort_field=docno&sort_direction=ASC&show_document_history=true
        doc_search_url=" https://apidev.aconex.com/api/projects/{}/register?search_query=filetype:\"{}\" AND Door&return_fields=approved,asBuiltRequired,attribute1,attribute2,attribute3,attribute4,author,authorisedBy,category&search_type=NUMBER_LIMITED&search_result_size=75&sort_field=docno&sort_direction=ASC&show_document_history=true"
        doc_search_url=doc_search_url.format(projectId, searchString)
        
        return self.getResponse(doc_search_url, self.headers, self.username, self.password)
            
    def documentIntegrity(self, projectId, sinceDate="2001-10-26T19:00:00.000Z"):
        
        #https://apidev.aconex.com/api/projects/{PROJECTID}/register/integrity?everythingsince=<2001-10-26T19:00:00.000Z>&show_document_history=true
        sincedate ="2001-10-26T19:00:00.000Z"
        audit_doc_url=" https://apidev.aconex.com/api/projects/{}/register/integrity?everythingsince={}&show_document_history=true"
        audit_doc_url=audit_doc_url.format(projectId, sincedate)
        
        return self.getResponse(self, audit_doc_url, self.headers, self.username, self.password)
    
    @staticmethod
    def getResponse(apiurl, headers, username, password):
             
        with requests.Session() as session:
            
            session.auth = (username, password)        
            # Instead of requests.get(), you'll use session.get()
            response = session.get(apiurl, headers=headers, verify=False)
            response.encoding ='utf-8'
            
        return response
    
    def formatxmlstring(self, uglyxml):
        
        xmlstr = xml.dom.minidom.parseString(uglyxml)
        xml_pretty_str = xmlstr.toprettyxml()
        
        return xml_pretty_str

        
def main():
    
    print("Hello World!")
    projectsurl = 'https://apidev.aconex.com/api/projects/'
    secretekey = '0c3d68fa-4348-4eee-8a1b-eff1c7b2f030'
    username = 'poleary'
    password='Auth3nt1c'
    
    sampleProjectid=1879048428
# =============================================================================
#     sampleTrackId=271341877549075742
# =============================================================================
    sampleDocId=271341877549084788
    
    downloadBasePath="D:\\EDMS\Aconex\\API-Data\\"
    
    parser = ET.XMLParser(encoding="utf-8")
    
    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]') 
    
    # Create Base folder if it doesnt exists
    try:
        os.mkdir(downloadBasePath)
        print("Directory " , downloadBasePath ,  " Created ") 
    except FileExistsError:
        print("Directory " , downloadBasePath ,  " already exists")
    
    
    apiclient = AconexAPIClient(projectsurl, secretekey, username, password)
    
    response = apiclient.loadProjects()
   
    content = response.text.encode('utf8')
    filename = Path('{}Aconex-{}.xml'.format(downloadBasePath, "Projects"))
    filename.write_text(apiclient.formatxmlstring(content)) 
    
    projectroot = ET.fromstring(content, parser=parser)
    #projectroot = ET.parse(filename).getroot()
    
    projectid_list=[]
    for eleprojectId in projectroot.iter('ProjectId'):
        projectid_list.append(eleprojectId.text)
    
    
    for pid in projectid_list:
        
              
        print("Processiong Project : " + pid)
        
        
        sampleProjectid = pid
        # Create Project ID folder to group all items for project
        projectPath = '{}\\{}\\'.format(downloadBasePath,sampleProjectid)
        try:
            os.mkdir(projectPath)
            print("Directory " , projectPath ,  " Created ") 
        except FileExistsError:
            print("Directory " , projectPath ,  " already exists")
        
# =============================================================================
#         response1 = apiclient.loadDocumentRegisterSchema(sampleProjectid)
#         #response1=response1.encode('utf8')
#         content1 = response1.text.encode('utf8')
#         filename1 = Path('{}\\{}.xml'.format(projectPath,"DocRegisterSchema"))
#         filename1.write_text(apiclient.formatxmlstring(content1))     
# =============================================================================
        
        response2 = apiclient.loadDocumentRegister(sampleProjectid)
        #response2=response2.encode('utf8')
        content2 = response2.text.encode('utf8')
        filename2 = Path('{}\\{}.xml'.format(projectPath, "DocRegister"))
        filename2.write_text(apiclient.formatxmlstring(content2)) 
        
        #docregisterroot = ET.fromstring(content2, parser=parser)
        docregisterroot = ET.parse(filename2).getroot()
        doc_and_tracking_id={}
        for document in docregisterroot.iter('Document'):
            docId =  document.get('DocumentId')
            trackId = document.find('TrackingId').text
            if docId not in doc_and_tracking_id.keys():
                doc_and_tracking_id[docId] = trackId
        
        for key in doc_and_tracking_id:
            
            print("Processiong Tracking : " + key)
            sampleDocId = key
# =============================================================================
#             sampleTrackId = doc_and_tracking_id[key]
# =============================================================================
            
            response3 = apiclient.loadDocumentMetadata(sampleProjectid, sampleDocId)
            #response3=response3.encode('utf8')
            content3 = response3.text.encode('utf8')
            filename3 = Path('{}\\{}-{}.xml'.format(projectPath,sampleDocId,"DocumentMetadata"))
            filename3.write_text(apiclient.formatxmlstring(content3)) 
            
            #docroot = ET.fromstring(content3, parser=parser)
            docroot = ET.parse(filename3).getroot()
            
            documentname='{}.{}'
            docnumber = docroot.find('DocumentNumber').text
            if(regex.search(docnumber) == None): 
                print("String is accepted")
            else:
                break
            
            filetype= docroot.find('FileType').text
            documentname = documentname.format(docnumber,filetype)
            print("File Name : " + documentname)
            
# =============================================================================
#             response4 = apiclient.loadDocumentVersions(sampleProjectid, sampleTrackId)
#             #response4=response4.encode('utf8')
#             content4 = response4.text.encode('utf8')
#             filename4 = Path('{}\\{}-{}.xml'.format(projectPath, sampleTrackId,"DocumentVersions"))
#             filename4.write_text(apiclient.formatxmlstring(content4)) 
# =============================================================================
            
            response5 = apiclient.downloadContentFile(sampleProjectid, sampleDocId)
            #response5=response5.encode('utf8')
            content5 = response5.content
            filename5 = Path('{}\\{}-{}.{}'.format(projectPath, sampleDocId, docnumber, filetype))
            filename5.write_bytes(content5) 
    
if __name__ == '__main__':
    main()

