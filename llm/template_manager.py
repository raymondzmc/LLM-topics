import jinja2


class JinjaTemplateManager:
    ''' Loads templates from chatbot_api/llm/prompts/templates and renders them with jinja2'''

    def __init__(self, template_dir: str = 'llm/templates'):
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir))

    def render(self, template_name: str, **kwargs) -> str:
        ''' Render a template with the given arguments'''

        template = self.env.get_template(template_name)
        rendered_text = template.render(**kwargs)
        return rendered_text


jinja_template_manager = JinjaTemplateManager()