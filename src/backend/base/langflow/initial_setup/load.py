from .starter_projects import (
    basic_prompting_graph,
    blog_writer_graph,
    document_qa_graph,
    memory_chatbot_graph,
    near_vector_store_rag_graph,
    simple_near_agent_graph,
)


def get_starter_projects_graphs():
    return [
        basic_prompting_graph(),
        blog_writer_graph(),
        document_qa_graph(),
        memory_chatbot_graph(),
        near_vector_store_rag_graph(),
        simple_near_agent_graph(),
    ]


def get_starter_projects_dump():
    return [g.dump() for g in get_starter_projects_graphs()]
