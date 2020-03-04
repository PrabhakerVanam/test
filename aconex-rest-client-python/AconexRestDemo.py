import collections
import inspect
import logging
import logging.config
import sys
import time
from builtins import input
from configparser import ConfigParser

from model import RestLink
from model.QueryDocument import Sort, ExpSet, FtExp, FacetDefinition
from network import RestClient
from network.RestClient import MEDIA_TYPE_DM_JSON
from util import ResourceUtility
from util.DemoUtil import format_json
from util.DemoUtil import print_properties
from util.DemoUtil import print_resource_properties

__author__ = 'Prabhaker Vanam'

DEMO_CABINET = 'demo-cabinet'
DEMO_NEW_FOLDER = 'demo-new-folder'
DEMO_UPDATE_FOLDER = 'demo-update-folder'
DEMO_TEMP_FOLDER = 'demo-temp-folder'
DEMO_NEW_SYSOBJECT = 'demo-new-sysobj'
DEMO_UPDATE_SYSOBJECT = 'demo-update-sysobj'
DEMO_NEW_DOCUMENT = 'demo-doc'
DEMO_CHECK_IN_WITHOUT_CONTENT = 'demo-check-in-without-content'
DEMO_CHECK_IN_WITH_CONTENT = 'demo-check-in-with-content'
DEMO_NEW_USER = 'demo-py-client-user'
DEMO_UPDATE_USER = 'demo-py-client-user-updated'
DEMO_NEW_GROUP = 'demo-py-client-group'
DEMO_ANOTHER_NEW_GROUP = 'demo-py-client-another-group'
DEMO_UPDATE_GROUP = 'demo-py-client-group-updated'
DEMO_SHARABLE_OBJECT = 'demo_sharable_obj'
DEMO_ANOTHER_SHARABLE_OBJECT = 'demo_another_sharable_obj'
DEMO_LIGHT_WEITHT_OBJECT = 'demo_lightweight_obj'
DEMO_OBJECT_TO_ATTACH = 'obj_to_attach'

VERSION_72 = '7.2'
VERSION_73 = '7.3'

logger = logging.getLogger(__name__)

Demo = collections.namedtuple('Demo', ['version', 'description', 'callable'])


class AconexRestDemo:
    def __init__(self, prompt):
        self.prompt = prompt

        self._init_logger()

        self._init_client()

        self._init_demo_items()

    def _init_demo_items(self):

        try:
            demos = tuple(
                Demo(self._get_method_version(method), self._get_method_doc(method), method) for name, method in
                (inspect.getmembers(self, predicate=inspect.ismethod))
                if str(name).startswith('demo'))

            product_version = float(self.client.get_product_info().get('properties').get('major'))

            print('This is Aconex REST {}'.format(product_version))
            self._populate_demo_items(demos, product_version)
        except Exception as e:
            logger.info('Error occurs... Quit demo.\n{}'.format(e))
            sys.exit(0)

    @staticmethod
    def _get_method_version(demo):
        return inspect.getdoc(demo).split('\n')[1].split(':')[1].strip()

    @staticmethod
    def _get_method_doc(demo):
        return inspect.getdoc(demo).split('\n')[0]

    def _populate_demo_items(self, demos, product_version):
        items = [demo
                 for demo in demos
                 if float(demo.version) <= product_version]

        # hardcode quit and reset at the top
        items.insert(0, Demo(self._get_method_version(self.quit), self._get_method_doc(self.quit), self.quit))
        items.insert(1, Demo(self._get_method_version(self.reset_environment),
                             self._get_method_doc(self.reset_environment), self.reset_environment))

        self.choices = {i: item
                        for i, item in enumerate(items)}

    def _init_client(self):
        config_parser = ConfigParser()
        config_parser.read("resources/rest.properties")

        self.REST_URI = config_parser.get("environment", "rest.host")
        rest_uri = self.prompt.rest_entry(self.REST_URI)
        if rest_uri:
            self.REST_URI = rest_uri

        self.REST_SECRETKEY = config_parser.get("environment", "rest.secretkey")
        rest_sec_key = self.prompt.rest_sec_key(self.REST_SECRETKEY)
        if rest_sec_key:
            self.REST_SECRETKEY = rest_sec_key

        self.REST_USER = config_parser.get("environment", "rest.username")
        rest_user = self.prompt.rest_user(self.REST_USER)
        if rest_user:
            self.REST_USER = rest_user

        self.REST_PWD = config_parser.get("environment", "rest.password")
        rest_pwd = self.prompt.rest_pwd(self.REST_PWD)
        if rest_pwd:
            self.REST_PWD = rest_pwd

        self.client = RestClient.RestClient(self.REST_USER, self.REST_PWD, self.REST_URI, self.REST_SECRETKEY)

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

    @staticmethod
    def quit():
        """Quit the demo  
        version: 1.2
        """

        logger.info("\nQuit the demo.")
        sys.exit(0)

    def create_demo_cabinet(self):
        logger.debug("\n+++++++++++++++++++++++++++++++Create temp cabinet Start+++++++++++++++++++++++++++++++")

        self.client.create_cabinet(ResourceUtility.generate_cabinet(object_name=DEMO_CABINET))

        logger.debug("+++++++++++++++++++++++++++++++Create temp cabinet End+++++++++++++++++++++++++++++++")

      
  
    def demo_dql(self):
        """
        REST DQL
        version: 7.2
        """
        logger.info("\n+++++++++++++++++++++++++++++++DQL Start+++++++++++++++++++++++++++++++")

        logger.info('Query \'select * from dm_user\' with items-per-page=3,page=2...')
        results = self.client.dql('select * from dm_user', {'items-per-page': '2', 'page': '2'})

        logger.info('Object names in page %d...', 2)
        for result in results.get_entries():
            logger.info(result.get('content').get('properties').get('user_name'))
        logger.info('')

        logger.info('Navigate to next page...')
        results = self.client.next_page(results)

        if results is None:
            logger.info('Next page does not exist.')
        else:
            logger.info('Object names in page %d...', 3)
            for result in results.get_entries():
                logger.info(result.get('content').get('properties').get('user_name'))

        logger.info("+++++++++++++++++++++++++++++++DQL End+++++++++++++++++++++++++++++++")

    def demo_simple_search(self):
        """
        REST search with URL parameters
        version:7.2
        """
        logger.info("\n+++++++++++++++++++++++++++++++Simple Search Start+++++++++++++++++++++++++++++++")

        logger.info('Simple search with keyword emc and parameters items-per-page=3,page=2,inline=true...')
        results = self.client.simple_search('emc', {'items-per-page': '2', 'page': '1', 'inline': 'true'})

        logger.info('Object names in page %d...', 2)
        for result in results.get_entries():
            logger.info(result.get('content').get('properties').get('object_name'))

        logger.info('Navigate to next page...')
        results = self.client.next_page(results)

        if results is None:
            logger.info('Next page does not exist.')
        else:
            logger.info('Object names in page %d...', 3)
            for result in results.get_entries():
                logger.info(result.get('content').get('properties').get('object_name'))

        logger.info("+++++++++++++++++++++++++++++++Simple Search End+++++++++++++++++++++++++++++++")

      

    def demo_search_template(self):
        """
        REST Search Template
        version:7.3
        """
        logger.info("\n+++++++++++++++++++++++++++++++Search Template Start+++++++++++++++++++++++++++++++")

        self.step_separator('Create a search template...')
        query_doc = ResourceUtility.generate_query_document(types=['ad_document'], columns=['object_name'],
                                                            sorts=[Sort('object_name', True, 'en', True)],
                                                            expression_set=ExpSet('AND', FtExp('emc or rest',
                                                                                               is_template=True)))
        new_search_template = ResourceUtility.generate_search_template('New search template',
                                                                       'This is a new search template for demo', True,
                                                                       query_doc=query_doc)

        search_template = self.client.create_search_template(new_search_template)
        properties = search_template.get('properties')
        logger.info('New search template is created at {}.\n name:{},\n description: {},\n is_public: {}'.format(
            properties.get('r_creation_date'), properties.get('object_name'), properties.get('subject'),
            properties.get('r_is_public')))

        logger.info('Get search templates...')
        search_templates = self.client.get_search_templates()

        if len(search_templates.get_entries()) > 0:
            logger.info('Saved searches: ')
            for search_template in search_templates.get_entries():
                logger.info(search_template.get('title'))

        self.step_separator('Get one search template "{0}"...'.format(search_templates.get_entry(0).get('title')))
        search_template = self.client.get_search_template(search_templates.get_entry(0).get('title'))
        properties = search_template.get('properties')
        logger.info(
            'name: {},\n public: {},\n description: {}'.format(properties.get('object_name'),
                                                               properties.get('r_is_public'),
                                                               properties.get('subject')))
        logger.info('\nexternal variables:')
        for v in search_template.get('external-variables'):
            logger.info(
                'id: {},\nvariable type: {},\ndata type: {},\nvalue: {}'.format(v.get('id'), v.get('variable-type'),
                                                                                v.get('data-type'),
                                                                                v.get('variable-value')))

        logger.info(
            'saved AQL is:\n {}'.format(
                format_json(search_template.get('query-document-template')), indent=4))

        self.step_separator('Execute the search template...')

        input_variables = ResourceUtility.generate_search_template_variables(search_template.get('external-variables'),
                                                                             self.prompt.search_template_var)

        results = self.client.execute_search_template(search_template, variables=input_variables,
                                                      params={'items-per-page': '2', 'page': '1', 'inline': 'true'})
        logger.info('Object names in page %d...', 2)
        for result in results.get_entries():
            logger.info(result.get('content').get('properties').get('object_name'))

        self.step_separator('Delete the search template...')
        self.client.delete(search_template)

        logger.info("\n+++++++++++++++++++++++++++++++Search Template End+++++++++++++++++++++++++++++++")

     
    @staticmethod
    def step_separator(message):
        logger.info('\n' + message)

  

    def run_all(self):
        self.prepare_env()
        [item.callable()
         for key, item in self.choices.items()
         if not (item.callable == self.quit or item.callable == self.reset_environment)]

        self.reset_environment()

    def run(self):

        while True:
            try:
                for k, v in self.choices.items():
                    print("%d. %s" % (k, v.description))

                user_choice = int(self.prompt.demo_choice())

                if user_choice not in self.choices:
                    print('#Invalid choice!#\n')
                    continue
                
                self.choices[user_choice].callable()
                
                time.sleep(1)
            except ValueError:
                print("\n#Enter number of the demo items instead of other characters.#\n")
            except Exception as e:
                logger.exception(e)
                time.sleep(1)
                print("\n#Error is detected during demo. Please refer the log for the exception detail.#\n")


class PromptUserInput(object):
    rest_entry_msg = 'Input Documentum REST Entry Path: [default - {}]'

    rest_secretkey_msg = 'Input Client Secret Key: [default - {}]'

    rest_user_msg = 'Input User Name: [default - {}]'

    rest_pwd_msg = 'Input User Password: [default - {}]'

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
        return self.prompt_func(self.rest_sec_key_msg.format(default_key))

    def rest_user(self, default_user):
        return self.prompt_func(self.rest_user_msg.format(default_user))

    def rest_pwd(self, default_pwd):
        return self.prompt_func(self.rest_pwd_msg.format(default_pwd))

    def demo_logging(self):
        return self.prompt_func(self.demo_logging_msg)

    def demo_choice(self):
        return self.prompt_func(self.demo_choice_msg)

    def search_template_var(self, var_id, var_value):
        return self.prompt_func(self.search_template_var_msg.format(var_id, var_value))


def main():
    AconexRestDemo(PromptUserInput()).run()


if __name__ == '__main__':
    main()
else:
    logger.info('RestDemo as a module')
