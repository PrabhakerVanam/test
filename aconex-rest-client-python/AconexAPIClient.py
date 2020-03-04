# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:33:33 2020

@author: ad1006362
"""
'''
Class wich connect to the Aconex Cloud solution to geth the Project workspace items and synchronizes with the DMS system

'''

import collections
#import inspect
import logging
import logging.config
#import sys
import time
from builtins import input
from configparser import ConfigParser

import requests

import xml.etree.ElementTree as ET

import os

import re 

from pathlib import Path

import xml.dom.minidom

logger = logging.getLogger(__name__)

Demo = collections.namedtuple('Demo', ['version', 'description', 'callable'])


class AconexAPIClient:
    
# =============================================================================
#     def __init__(self, projectsurl, secretkey, username, password):
# =============================================================================
    def __init__(self, prompt):
        
        self.prompt = prompt
        
        self._init_logger()
        
        self._init_client()
        
        self.projectsurl = self.REST_URI
        self.headers = {'X-Application-Key': self.REST_SECRETKEY}
        self.username = self.REST_USER
        self.password = self.REST_PWD
        
# =============================================================================
#         self.projectsurl = projectsurl
#         self.headers = {'X-Application-Key': secretkey}
#         self.username = username
#         self.password = password
# =============================================================================
    
    def _init_client(self):
        config_parser = ConfigParser()
        config_parser.read("resources/rest.properties")

        self.REST_URI = config_parser.get("environment", "rest.host")
        rest_uri = self.prompt.rest_entry(self.REST_URI)
        if rest_uri:
            self.REST_URI = rest_uri

        self.REST_SECRETKEY = config_parser.get("environment", "rest.secretkey")
        rest_secretkey = self.prompt.rest_secretkey(self.REST_SECRETKEY)
        if rest_secretkey:
            self.REST_SECRETKEY = rest_secretkey

        self.REST_USER = config_parser.get("environment", "rest.username")
        rest_user = self.prompt.rest_user(self.REST_USER)
        if rest_user:
            self.REST_USER = rest_user

        self.REST_PWD = config_parser.get("environment", "rest.password")
        rest_pwd = self.prompt.rest_pwd(self.REST_PWD)
        if rest_pwd:
            self.REST_PWD = rest_pwd
            
        self.REST_DOWNLOAD_BASE_PATH = config_parser.get("environment", "rest.downloadpath")
        rest_download_path = self.prompt.rest_download_path(self.REST_DOWNLOAD_BASE_PATH)
        if rest_download_path:
            self.REST_DOWNLOAD_BASE_PATH = rest_download_path
        
        #self.client = RestClient.RestClient(self.REST_USER, self.REST_PWD, self.REST_URI, self.REST_SECRETKEY)
        
    
    def _init_logger(self):
        logging.getLogger("requests").setLevel(logging.WARNING)

        is_debug = self.prompt.demo_logging()
        if is_debug == 'yes':
            level = 'DEBUG'
        else:
            level = 'INFO'
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,

            'handlers': {
                'default': {
                    'level': level,
                    'class': 'logging.StreamHandler',
                },
            },
            'loggers': {
                '': {
                    'handlers': ['default'],
                    'level': 'DEBUG',
                    'propagate': True
                }
            }
        })
    
    
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
    
    def run(self):

        sampleProjectid=1879048428
        sampleTrackId=271341877549075742
        sampleDocId=271341877549084788
        
         #apiclient = AconexAPIClient(projectsurl, secretkey, username, password)
        apiclient = AconexAPIClient(self.prompt)
        
        #downloadBasePath="D:\\EDMS\\Aconex\\API-Data\\"    
        downloadBasePath=apiclient.REST_DOWNLOAD_BASE_PATH
        # Create Base folder if it doesnt exists
        try:
            os.mkdir(downloadBasePath)
            print("Directory " , downloadBasePath ,  " Created ") 
        except FileExistsError:
            print("Directory " , downloadBasePath ,  " already exists")
        
           
        response = apiclient.loadProjects()
        content = response.text
        filename = Path('{}Aconex-{}.xml'.format(downloadBasePath, "Projects"))
        filename.write_text(apiclient.formatxmlstring(content))  
        
            
        # Create Project ID folder to group all items for project
        projectPath = '{}\\{}\\'.format(downloadBasePath,sampleProjectid)
        try:
            os.mkdir(projectPath)
            print("Directory " , projectPath ,  " Created ") 
        except FileExistsError:
            print("Directory " , projectPath ,  " already exists")
            
        
        response1 = apiclient.loadDocumentRegisterSchema(sampleProjectid)
        content1 = response1.text
        filename1 = Path('{}\\{}.xml'.format(projectPath,"DocRegisterSchema"))
        filename1.write_text(apiclient.formatxmlstring(content1))     
        
        response2 = apiclient.loadDocumentRegister(sampleProjectid)
        content2 = response2.text
        filename2 = Path('{}\\{}.xml'.format(projectPath, "DocRegister"))
        filename2.write_text(apiclient.formatxmlstring(content2)) 
        
        response3 = apiclient.loadDocumentMetadata(sampleProjectid, sampleDocId)
        content3 = response3.text
        filename3 = Path('{}\\{}-{}.xml'.format(projectPath,sampleDocId,"DocumentMetadata"))
        filename3.write_text(apiclient.formatxmlstring(content3)) 
        
        response4 = apiclient.loadDocumentVersions(sampleProjectid, sampleTrackId)
        content4 = response4.text
        filename4 = Path('{}\\{}-{}.xml'.format(projectPath, sampleTrackId,"DocumentVersions"))
        filename4.write_text(apiclient.formatxmlstring(content4)) 
        
        response5 = apiclient.downloadContentFile(sampleProjectid, sampleDocId)
        content5 = response5.content
        filename5 = Path('{}\\{}.{}'.format(projectPath, sampleDocId, "pdf"))
        filename5.write_bytes(content5) 
    
       
    

class PromptUserInput(object):
    rest_entry_msg = 'Input Aconex REST Entry Path: [default - {}]'

    rest_secretkey_msg = 'Input Client Secret Key: [default - {}]'

    rest_user_msg = 'Input User Name: [default - {}]'

    rest_pwd_msg = 'Input User Password: [default - {}]'
    
    rest_download_path_msg = 'Input Default Download content Path: [default - {}]'

    demo_logging_msg = 'Enable debugging messages (yes|no)? [default - no]'

    demo_choice_msg = '\nWhat\'s your choice?\n'  

    search_template_var_msg = 'Input value for variable {}={}: '


    @staticmethod
    def prompt_func(message):
        time.sleep(0.2)
        return input(message)

    def rest_entry(self, default_entry):
        return self.prompt_func(self.rest_entry_msg.format(default_entry))

    def rest_secretkey(self, default_key):
        return self.prompt_func(self.rest_secretkey_msg.format(default_key))

    def rest_user(self, default_user):
        return self.prompt_func(self.rest_user_msg.format(default_user))

    def rest_pwd(self, default_pwd):
        return self.prompt_func(self.rest_pwd_msg.format(default_pwd))
    
    def rest_download_path(self, default_path):
        return self.prompt_func(self.rest_download_path_msg.format(default_path))

    def demo_logging(self):
        return self.prompt_func(self.demo_logging_msg)

    def demo_choice(self):
        return self.prompt_func(self.demo_choice_msg)

    def search_template_var(self, var_id, var_value):
        return self.prompt_func(self.search_template_var_msg.format(var_id, var_value))

        
def dummymain():
    
    print("Hello World!")
#    projectsurl = 'https://apidev.aconex.com/api/projects/'
#    secretkey = '0c3d68fa-4348-4eee-8a1b-eff1c7b2f030'
#    username = 'poleary'
#    password='Auth3nt1c'
    
    #sampleProjectid=1879048428
    #sampleTrackId=271341877549075742
    #sampleDocId=271341877549084788
    
     #apiclient = AconexAPIClient(projectsurl, secretkey, username, password)
    apiclient = AconexAPIClient()
    
    #downloadBasePath="D:\\EDMS\Aconex\\API-Data\\"    
    downloadBasePath=apiclient.REST_DOWNLOAD_BASE_PATH
    
    parser = ET.XMLParser(encoding="utf-8")
    
    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]') 
    
    # Create Base folder if it doesnt exists
    try:
        os.mkdir(downloadBasePath)
        print("Directory " , downloadBasePath ,  " Created ") 
    except FileExistsError:
        print("Directory " , downloadBasePath ,  " already exists")
    
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
    

def main():
    AconexAPIClient(PromptUserInput()).run()


if __name__ == '__main__':
    main()
else:
    logger.info('AconexAPIClient as a module')

