try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("Import successful")
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    print("Initialization successful")
    
    res = llm.invoke("Hello")
    print("Invoke successful")
    print(res.content)

except Exception as e:
    import traceback
    traceback.print_exc()
