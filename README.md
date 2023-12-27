# LangChain 中文入門教程

> 為了便於閱讀，已生成gitbook：[https://liaokong.gitbook.io/llm-kai-fa-jiao-cheng/](https://liaokong.gitbook.io/llm-kai-fa-jiao-cheng/)
>
> github地址：[https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide](https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide)

> 因為langchain庫一直在飛速更新迭代，但該文檔寫與4月初，並且我個人精力有限，所以colab裡面的代碼有可能有些已經過時。如果有運行失敗的可以先搜索一下當前文檔是否有更新，如文檔也沒更新歡迎提issue，或者修復後直接提pr，感謝~

> 加了個 [CHANGELOG](CHANGELOG.md),更新了新的內容我會寫在這裡，方便之前看過的朋友快速查看新的更新內容

> 如果想把 OPENAI API 的請求根路由修改成自己的代理地址，可以通過設置環境變量 「OPENAI\_API\_BASE」 來進行修改。
>
> 相關參考代碼：[https://github.com/openai/openai-python/blob/d6fa3bfaae69d639b0dd2e9251b375d7070bbef1/openai/\_\_init\_\_.py#L48](https://github.com/openai/openai-python/blob/d6fa3bfaae69d639b0dd2e9251b375d7070bbef1/openai/\_\_init\_\_.py#L48)
>
> 或在初始化OpenAI相關模型對像時，傳入「openai\_api\_base」 變量。
>
> 相關參考代碼：[https://github.com/hwchase17/langchain/blob/master/langchain/llms/openai.py#L148](https://github.com/hwchase17/langchain/blob/master/langchain/llms/openai.py#L148)

## 介紹

眾所周知 OpenAI 的 API 無法聯網的，所以如果只使用自己的功能實現聯網搜索並給出回答、總結 PDF 文檔、基於某個 Youtube 視頻進行問答等等的功能肯定是無法實現的。所以，我們來介紹一個非常強大的第三方開源庫：`LangChain` 。

> 文檔地址：https://python.langchain.com/en/latest/

這個庫目前非常活躍，每天都在迭代，已經有 22k 的 star，更新速度飛快。

LangChain 是一個用於開發由語言模型驅動的應用程序的框架。他主要擁有 2 個能力：

1. 可以將 LLM 模型與外部數據源進行連接
2. 允許與 LLM 模型進行交互

> LLM 模型：Large Language Model，大型語言模型

##

## 基礎功能

LLM 調用

* 支持多種模型接口，比如 OpenAI、Hugging Face、AzureOpenAI ...
* Fake LLM，用於測試
* 緩存的支持，比如 in-mem（內存）、SQLite、Redis、SQL
* 用量記錄
* 支持流模式（就是一個字一個字的返回，類似打字效果）

Prompt管理，支持各種自定義模板

擁有大量的文檔加載器，比如 Email、Markdown、PDF、Youtube ...

對索引的支持

* 文檔分割器
* 向量化
* 對接向量存儲與搜索，比如 Chroma、Pinecone、Qdrand

Chains

* LLMChain
* 各種工具Chain
* LangChainHub

## 必知概念

相信大家看完上面的介紹多半會一臉懵逼。不要擔心，上面的概念其實在剛開始學的時候不是很重要，當我們講完後面的例子之後，在回來看上面的內容會一下明白很多。

但是，這裡有幾個概念是必須知道的。

##

### Loader 加載器

顧名思義，這個就是從指定源進行加載數據的。比如：文件夾 `DirectoryLoader`、Azure 存儲 `AzureBlobStorageContainerLoader`、CSV文件 `CSVLoader`、印象筆記 `EverNoteLoader`、Google網盤 `GoogleDriveLoader`、任意的網頁 `UnstructuredHTMLLoader`、PDF `PyPDFLoader`、S3 `S3DirectoryLoader`/`S3FileLoader`、

Youtube `YoutubeLoader` 等等，上面只是簡單的進行列舉了幾個，官方提供了超級的多的加載器供你使用。

> https://python.langchain.com/docs/modules/data_connection/document_loaders.html

###

### Document 文檔

當使用loader加載器讀取到數據源後，數據源需要轉換成 Document 對像後，後續才能進行使用。

###

### Text Spltters 文本分割

顧名思義，文本分割就是用來分割文本的。為什麼需要分割文本？因為我們每次不管是做把文本當作 prompt 發給 openai api ，還是還是使用 openai api embedding 功能都是有字符限制的。

比如我們將一份300頁的 pdf 發給 openai api，讓他進行總結，他肯定會報超過最大 Token 錯。所以這裡就需要使用文本分割器去分割我們 loader 進來的 Document。

###

### Vectorstores 向量數據庫

因為數據相關性搜索其實是向量運算。所以，不管我們是使用 openai api embedding 功能還是直接通過向量數據庫直接查詢，都需要將我們的加載進來的數據 `Document` 進行向量化，才能進行向量運算搜索。轉換成向量也很簡單，只需要我們把數據存儲到對應的向量數據庫中即可完成向量的轉換。

官方也提供了很多的向量數據庫供我們使用。

> https://python.langchain.com/en/latest/modules/indexes/vectorstores.html

###

### Chain 鏈

我們可以把 Chain 理解為任務。一個 Chain 就是一個任務，當然也可以像鏈條一樣，一個一個的執行多個鏈。

###

### Agent 代理

我們可以簡單的理解為他可以動態的幫我們選擇和調用chain或者已有的工具。

執行過程可以參考下面這張圖:

![image-20230406213322739](doc/image-20230406213322739.png)

### Embedding

用於衡量文本的相關性。這個也是 OpenAI API 能實現構建自己知識庫的關鍵所在。

他相比 fine-tuning 最大的優勢就是，不用進行訓練，並且可以實時添加新的內容，而不用加一次新的內容就訓練一次，並且各方面成本要比 fine-tuning 低很多。

> 具體比較和選擇可以參考這個視頻：https://www.youtube.com/watch?v=9qq6HTr7Ocw

##

## 實戰

通過上面的必備概念大家應該已經可以對 LangChain 有了一定的瞭解，但是可能還有有些懵。

這都是小問題，我相信看完後面的實戰，你們就會徹底的理解上面的內容，並且能感受到這個庫的真正強大之處。

因為我們 OpenAI API 進階，所以我們後面的範例使用的 LLM 都是以Open AI 為例，後面大家可以根據自己任務的需要換成自己需要的 LLM 模型即可。

當然，在這篇文章的末尾，全部的全部代碼都會被保存為一個 colab 的 ipynb 文件提供給大家來學習。

> 建議大家按順序去看每個例子，因為下一個例子會用到上一個例子裡面的知識點。
>
> 當然，如果有看不懂的也不用擔心，可以繼續往後看，第一次學習講究的是不求甚解。

###

### 完成一次問答

第一個案例，我們就來個最簡單的，用 LangChain 加載 OpenAI 的模型，並且完成一次問答。

在開始之前，我們需要先設置我們的 openai 的 key，這個 key 可以在用戶管理裡面創建，這裡就不細說了。

```python
import os
os.environ["OPENAI_API_KEY"] = '你的api key'
```

然後，我們進行導入和執行

```py
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003",max_tokens=1024)
llm("怎麼評價人工智能")
```

![image-20230404232621517](doc/image-20230404232621517.png)

這時，我們就可以看到他給我們的返回結果了，怎麼樣，是不是很簡單。

### 通過 Google 搜索並返回答案

接下來，我們就來搞點有意思的。我們來讓我們的 OpenAI api 聯網搜索，並返回答案給我們。

這裡我們需要借助 Serpapi 來進行實現，Serpapi 提供了 google 搜索的 api 接口。

首先需要我們到 Serpapi 官網上註冊一個用戶，https://serpapi.com/ 並複製他給我們生成 api key。

然後我們需要像上面的 openai api key 一樣設置到環境變量裡面去。

```python
import os
os.environ["OPENAI_API_KEY"] = '你的api key'
os.environ["SERPAPI_API_KEY"] = '你的api key'
```

然後，開始編寫我的代碼

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType

# 加載 OpenAI 模型
llm = OpenAI(temperature=0,max_tokens=2048) 

 # 加載 serpapi 工具
tools = load_tools(["serpapi"])

# 如果搜索完想再計算一下可以這麼寫
# tools = load_tools(['serpapi', 'llm-math'], llm=llm)

# 如果搜索完想再讓他再用python的print做點簡單的計算，可以這樣寫
# tools=load_tools(["serpapi","python_repl"])

# 工具加載後都需要初始化，verbose 參數為 True，會打印全部的執行詳情
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 運行 agent
agent.run("What's the date today? What great events have taken place today in history?")
```

![image-20230404234236982](doc/image-20230404234236982.png)

我們可以看到，他正確的返回了日期（有時差），並且返回了歷史上的今天。

在 chain 和 agent 對像上都會有 `verbose` 這個參數，這個是個非常有用的參數，開啟他後我們可以看到完整的 chain 執行過程。

可以在上面返回的結果看到，他將我們的問題拆分成了幾個步驟，然後一步一步得到最終的答案。

關於agent type 幾個選項的含義（理解不了也不會影響下面的學習，用多了自然理解了）：

* zero-shot-react-description: 根據工具的描述和請求內容的來決定使用哪個工具（最常用）
* react-docstore: 使用 ReAct 框架和 docstore 交互, 使用`Search` 和`Lookup` 工具, 前者用來搜, 後者尋找term, 舉例: `Wipipedia` 工具
* self-ask-with-search 此代理只使用一個工具: Intermediate Answer, 它會為問題尋找事實答案(指的非 gpt 生成的答案, 而是在網絡中,文本中已存在的), 如 `Google search API` 工具
* conversational-react-description: 為會話設置而設計的代理, 它的prompt會被設計的具有會話性, 且還是會使用 ReAct 框架來決定使用來個工具, 並且將過往的會話交互存入內存

> reAct 介紹可以看這個：https://arxiv.org/pdf/2210.03629.pdf
>
> LLM 的 ReAct 模式的 Python 實現: https://til.simonwillison.net/llms/python-react-pattern
>
> agent type 官方解釋：
>
> https://python.langchain.com/en/latest/modules/agents/agents/agent_types.html?highlight=zero-shot-react-description

> 有一點要說明的是，這個 `serpapi` 貌似對中文不是很友好，所以提問的 prompt 建議使用英文。

當然，官方已經寫好了 `ChatGPT Plugins` 的 agent，未來 chatgpt 能用啥插件，我們在 api 裡面也能用插件，想想都美滋滋。

不過目前只能使用不用授權的插件，期待未來官方解決這個。

感興趣的可以看這個文檔：https://python.langchain.com/en/latest/modules/agents/tools/examples/chatgpt_plugins.html

> Chatgpt 只能給官方賺錢，而 Openai API 能給我賺錢

### 對超長文本進行總結

假如我們想要用 openai api 對一個段文本進行總結，我們通常的做法就是直接發給 api 讓他總結。但是如果文本超過了 api 最大的 token 限制就會報錯。

這時，我們一般會進行對文章進行分段，比如通過 tiktoken 計算並分割，然後將各段發送給 api 進行總結，最後將各段的總結再進行一個全部的總結。

如果，你用是 LangChain，他很好的幫我們處理了這個過程，使得我們編寫代碼變的非常簡單。

廢話不多說，直接上代碼。

```python
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI

# 導入文本
loader = UnstructuredFileLoader("/content/sample_data/data/lg_test.txt")
# 將文本轉成 Document 對像
document = loader.load()
print(f'documents:{len(document)}')

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 0
)

# 切分文本
split_documents = text_splitter.split_documents(document)
print(f'documents:{len(split_documents)}')

# 加載 llm 模型
llm = OpenAI(model_name="text-davinci-003", max_tokens=1500)

# 創建總結鏈
chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# 執行總結鏈，（為了快速演示，只總結前5段）
chain.run(split_documents[:5])
```

首先我們對切割前和切割後的 document 個數進行了打印，我們可以看到，切割前就是只有整篇的一個 document，切割完成後，會把上面一個 document 切成 317 個 document。

![image-20230405162631460](doc/image-20230405162631460.png)

最終輸出了對前 5 個 document 的總結。

![image-20230405162937249](doc/image-20230405162937249.png)

這裡有幾個參數需要注意：

**文本分割器的 `chunk_overlap` 參數**

這個是指切割後的每個 document 裡包含幾個上一個 document 結尾的內容，主要作用是為了增加每個 document 的上下文關聯。比如，`chunk_overlap=0`時， 第一個 document 為 aaaaaa，第二個為 bbbbbb；當 `chunk_overlap=2` 時，第一個 document 為 aaaaaa，第二個為 aabbbbbb。

不過，這個也不是絕對的，要看所使用的那個文本分割模型內部的具體算法。

> 文本分割器可以參考這個文檔：https://python.langchain.com/en/latest/modules/indexes/text_splitters.html

**chain 的 `chain_type` 參數**

這個參數主要控制了將 document 傳遞給 llm 模型的方式，一共有 4 種方式：

`stuff`: 這種最簡單粗暴，會把所有的 document 一次全部傳給 llm 模型進行總結。如果document很多的話，勢必會報超出最大 token 限制的錯，所以總結文本的時候一般不會選中這個。

`map_reduce`: 這個方式會先將每個 document 進行總結，最後將所有 document 總結出的結果再進行一次總結。

![image-20230405165752743](doc/image-20230405165752743.png)

`refine`: 這種方式會先總結第一個 document，然後在將第一個 document 總結出的內容和第二個 document 一起發給 llm 模型在進行總結，以此類推。這種方式的好處就是在總結後一個 document 的時候，會帶著前一個的 document 進行總結，給需要總結的 document 添加了上下文，增加了總結內容的連貫性。

![image-20230405170617383](doc/image-20230405170617383.png)

`map_rerank`: 這種一般不會用在總結的 chain 上，而是會用在問答的 chain 上，他其實是一種搜索答案的匹配方式。首先你要給出一個問題，他會根據問題給每個 document 計算一個這個 document 能回答這個問題的概率分數，然後找到分數最高的那個 document ，在通過把這個 document 轉化為問題的 prompt 的一部分（問題+document）發送給 llm 模型，最後 llm 模型返回具體答案。

### 構建本地知識庫問答機器人

在這個例子中，我們會介紹如何從我們本地讀取多個文檔構建知識庫，並且使用 Openai API 在知識庫中進行搜索並給出答案。

這個是個很有用的教程，比如可以很方便的做一個可以介紹公司業務的機器人，或是介紹一個產品的機器人。

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

# 加載文件夾中的所有txt類型的文件
loader = DirectoryLoader('/content/sample_data/data/', glob='**/*.txt')
# 將數據轉成 document 對象，每個文件會作為一個 document
documents = loader.load()

# 初始化加載器
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# 切割加載的 document
split_docs = text_splitter.split_documents(documents)

# 初始化 openai 的 embeddings 對像
embeddings = OpenAIEmbeddings()
# 將 document 通過 openai 的 embeddings 對像計算 embedding 向量信息並臨時存入 Chroma 向量數據庫，用於後續匹配查詢
docsearch = Chroma.from_documents(split_docs, embeddings)

# 創建問答對像
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
# 進行問答
result = qa({"query": "科大訊飛今年第一季度收入是多少？"})
print(result)
```

![image-20230405173730382](doc/image-20230405173730382.png)

我們可以通過結果看到，他成功的從我們的給到的數據中獲取了正確的答案。

> 關於 Openai embeddings 詳細資料可以參看這個連接: https://platform.openai.com/docs/guides/embeddings

### 構建向量索引數據庫

我們上個案例裡面有一步是將 document 信息轉換成向量信息和embeddings的信息並臨時存入 Chroma 數據庫。

因為是臨時存入，所以當我們上面的代碼執行完成後，上面的向量化後的數據將會丟失。如果想下次使用，那麼就還需要再計算一次embeddings，這肯定不是我們想要的。

那麼，這個案例我們就來通過 Chroma 和 Pinecone 這兩個數據庫來講一下如何做向量數據持久化。

> 因為 LangChain 支持的數據庫有很多，所以這裡就介紹兩個用的比較多的，更多的可以參看文檔:https://python.langchain.com/en/latest/modules/indexes/vectorstores/getting\_started.html

**Chroma**

chroma 是個本地的向量數據庫，他提供的一個 `persist_directory` 來設置持久化目錄進行持久化。讀取時，只需要調取 `from_document` 方法加載即可。

```python
from langchain.vectorstores import Chroma

# 持久化數據
docsearch = Chroma.from_documents(documents, embeddings, persist_directory="D:/vector_store")
docsearch.persist()

# 加載數據
docsearch = Chroma(persist_directory="D:/vector_store", embedding_function=embeddings)

```

**Pinecone**

Pinecone 是一個在線的向量數據庫。所以，我可以第一步依舊是註冊，然後拿到對應的 api key。https://app.pinecone.io/

> 免費版如果索引14天不使用會被自動清除。

然後創建我們的數據庫：

Index Name：這個隨意

Dimensions：OpenAI 的 text-embedding-ada-002 模型為 OUTPUT DIMENSIONS 為 1536，所以我們這裡填 1536

Metric：可以默認為 cosine

選擇starter plan

![image-20230405184646314](doc/starter-plan.png)

持久化數據和加載數據代碼如下

```python
# 持久化數據
docsearch = Pinecone.from_texts([t.page_content for t in split_docs], embeddings, index_name=index_name)

# 加載數據
docsearch = Pinecone.from_existing_index(index_name, embeddings)
```

一個簡單從數據庫獲取 embeddings，並回答的代碼如下

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import pinecone

# 初始化 pinecone
pinecone.init(
  api_key="你的api key",
  environment="你的Environment"
)

loader = DirectoryLoader('/content/sample_data/data/', glob='**/*.txt')
# 將數據轉成 document 對象，每個文件會作為一個 document
documents = loader.load()

# 初始化加載器
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# 切割加載的 document
split_docs = text_splitter.split_documents(documents)

index_name="liaokong-test"

# 持久化數據
# docsearch = Pinecone.from_texts([t.page_content for t in split_docs], embeddings, index_name=index_name)

# 加載數據
docsearch = Pinecone.from_existing_index(index_name,embeddings)

query = "科大訊飛今年第一季度收入是多少？"
docs = docsearch.similarity_search(query, include_metadata=True)

llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
chain.run(input_documents=docs, question=query)
```

![image-20230407001803057](doc/image-20230407001803057.png)

### 使用GPT3.5模型構建油管頻道問答機器人

在 chatgpt api（也就是 GPT-3.5-Turbo）模型出來後，因錢少活好深受大家喜愛，所以 LangChain 也加入了專屬的鏈和模型，我們來跟著這個例子看下如何使用他。

```python
import os

from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate
)

# 加載 youtube 頻道
loader = YoutubeLoader.from_youtube_url('https://www.youtube.com/watch?v=Dj60HHy-Kqk')
# 將數據轉成 document
documents = loader.load()

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=20
)

# 分割 youtube documents
documents = text_splitter.split_documents(documents)

# 初始化 openai embeddings
embeddings = OpenAIEmbeddings()

# 將數據存入向量存儲
vector_store = Chroma.from_documents(documents, embeddings)
# 通過向量存儲初始化檢索器
retriever = vector_store.as_retriever()

system_template = """
Use the following context to answer the user's question.
If you don't know the answer, say you don't, don't try to make it up. And answer in Chinese.
-----------
{question}
-----------
{chat_history}
"""

# 構建初始 messages 列表，這裡可以理解為是 openai 傳入的 messages 參數
messages = [
  SystemMessagePromptTemplate.from_template(system_template),
  HumanMessagePromptTemplate.from_template('{question}')
]

# 初始化 prompt 對像
prompt = ChatPromptTemplate.from_messages(messages)


# 初始化問答鏈
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1,max_tokens=2048),retriever,condense_question_prompt=prompt)


chat_history = []
while True:
  question = input('問題：')
  # 開始發送問題 chat_history 為必須參數,用於存儲對話歷史
  result = qa({'question': question, 'chat_history': chat_history})
  chat_history.append((question, result['answer']))
  print(result['answer'])
```

我們可以看到他能很準確的圍繞這個油管視頻進行問答

![image-20230406211923672](doc/image-20230406211923672.png)

使用流式回答也很方便

```python
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

chat = ChatOpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
resp = chat(chat_prompt_with_values.to_messages())
```

### 用 OpenAI 連接萬種工具

我們主要是結合使用 `zapier` 來實現將萬種工具連接起來。

所以我們第一步依舊是需要申請賬號和他的自然語言 api key。https://zapier.com/l/natural-language-actions

他的 api key 雖然需要填寫信息申請。但是基本填入信息後，基本可以秒在郵箱裡看到審核通過的郵件。

然後，我們通過右鍵裡面的連接打開我們的api 配置頁面。我們點擊右側的 `Manage Actions` 來配置我們要使用哪些應用。

我在這裡配置了 Gmail 讀取和發郵件的 action，並且所有字段都選的是通過 AI 猜。

![image-20230406233319250](doc/image-20230406233319250.png)

![image-20230406234827815](doc/image-20230406234827815.png)

配置好後，我們開始寫代碼

```python
import os
os.environ["ZAPIER_NLA_API_KEY"] = ''
```

```python
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper


llm = OpenAI(temperature=.3)
zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(toolkit.get_tools(), llm, agent="zero-shot-react-description", verbose=True)

# 我們可以通過打印的方式看到我們都在 Zapier 裡面配置了哪些可以用的工具
for tool in toolkit.get_tools():
  print (tool.name)
  print (tool.description)
  print ("\n\n")

agent.run('請用中文總結最後一封"******@qq.com"發給我的郵件。並將總結髮送給"******@qq.com"')
```

![image-20230406234712909](doc/image-20230406234712909.png)

我們可以看到他成功讀取了`******@qq.com`給他發送的最後一封郵件，並將總結的內容又發送給了`******@qq.com`

這是我發送給 Gmail 的郵件。

![image-20230406234017369](doc/image-20230406234017369.png)

這是他發送給 QQ 郵箱的郵件。

![image-20230406234800632](doc/image-20230406234800632.png)

這只是個小例子，因為 `zapier` 有數以千計的應用，所以我們可以輕鬆結合 openai api 搭建自己的工作流。

## 小例子們

一些比較大的知識點都已經講完了，後面的內容都是一些比較有趣的小例子，當作拓展延伸。

### **執行多個chain**

因為他是鏈式的，所以他也可以按順序依次去執行多個 chain

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

# location 鏈
llm = OpenAI(temperature=1)
template = """Your job is to come up with a classic dish from the area that the users suggests.
% USER LOCATION
{user_location}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_location"], template=template)
location_chain = LLMChain(llm=llm, prompt=prompt_template)

# meal 鏈
template = """Given a meal, give a short and simple recipe on how to make that dish at home.
% MEAL
{user_meal}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)
meal_chain = LLMChain(llm=llm, prompt=prompt_template)

# 通過 SimpleSequentialChain 串聯起來，第一個答案會被替換第二箇中的user_meal，然後再進行詢問
overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)
review = overall_chain.run("Rome")
```

![image-20230406000133339](doc/image-20230406000133339.png)

### **結構化輸出**

有時候我們希望輸出的內容不是文本，而是像 json 那樣結構化的數據。

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003")

# 告訴他我們生成的內容需要哪些字段，每個字段類型式啥
response_schemas = [
    ResponseSchema(name="bad_string", description="This a poorly formatted user input string"),
    ResponseSchema(name="good_string", description="This is your response, a reformatted response")
]

# 初始化解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 生成的格式提示符
# {
#	"bad_string": string  // This a poorly formatted user input string
#	"good_string": string  // This is your response, a reformatted response
#}
format_instructions = output_parser.get_format_instructions()

template = """
You will be given a poorly formatted string from a user.
Reformat it and make sure all the words are spelled correctly

{format_instructions}

% USER INPUT:
{user_input}

YOUR RESPONSE:
"""

# 將我們的格式描述嵌入到 prompt 中去，告訴 llm 我們需要他輸出什麼樣格式的內容
prompt = PromptTemplate(
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
    template=template
)

promptValue = prompt.format(user_input="welcom to califonya!")
llm_output = llm(promptValue)

# 使用解析器進行解析生成的內容
output_parser.parse(llm_output)
```

![image-20230406000017276](doc/image-20230406000017276.png)

### **爬取網頁並輸出JSON數據**

有些時候我們需要爬取一些<mark style="color:red;">**結構性比較強**</mark>的網頁，並且需要將網頁中的信息以JSON的方式返回回來。

我們就可以使用 `LLMRequestsChain` 類去實現，具體可以參考下面代碼

> 為了方便理解，我在例子中直接使用了Prompt的方法去格式化輸出結果，而沒用使用上個案例中用到的 `StructuredOutputParser`去格式化，也算是提供了另外一種格式化的思路

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """在 >>> 和 <<< 之間是網頁的返回的HTML內容。
網頁是新浪財經A股上市公司的公司簡介。
請抽取參數請求的信息。

>>> {requests_result} <<<
請使用如下的JSON格式返回數據
{{
  "company_name":"a",
  "company_english_name":"b",
  "issue_price":"c",
  "date_of_establishment":"d",
  "registered_capital":"e",
  "office_address":"f",
  "Company_profile":"g"

}}
Extracted:"""

prompt = PromptTemplate(
    input_variables=["requests_result"],
    template=template
)

chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=prompt))
inputs = {
  "url": "https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/600519.phtml"
}

response = chain(inputs)
print(response['output'])
```

我們可以看到，他很好的將格式化後的結果輸出了出來

<figure><img src="doc/image-20230510234934.png" alt=""><figcaption></figcaption></figure>

### **自定義agent中所使用的工具**

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper

llm = OpenAI(temperature=0)

# 初始化搜索鏈和計算鏈
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

# 創建一個功能列表，指明這個 agent 裡面都有哪些可用工具，agent 執行過程可以看必知概念裡的 Agent 那張圖
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )
]

# 初始化 agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 執行 agent
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")

```

![image-20230406002117283](doc/image-20230406002117283.png)

自定義工具裡面有個比較有意思的地方，使用哪個工具的權重是靠 `工具中描述內容` 來實現的，和我們之前編程靠數值來控制權重完全不同。

比如 Calculator 在描述裡面寫到，如果你問關於數學的問題就用他這個工具。我們就可以在上面的執行過程中看到，他在我們請求的 prompt 中數學的部分，就選用了Calculator 這個工具進行計算。

### **使用Memory實現一個帶記憶的對話機器人**

上一個例子我們使用的是通過自定義一個列表來存儲對話的方式來保存歷史的。

當然，你也可以使用自帶的 memory 對像來實現這一點。

```python
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0)

# 初始化 MessageHistory 對像
history = ChatMessageHistory()

# 給 MessageHistory 對像添加對話內容
history.add_ai_message("你好！")
history.add_user_message("中國的首都是哪裡？")

# 執行對話
ai_response = chat(history.messages)
print(ai_response)
```

### **使用 Hugging Face 模型**

使用 Hugging Face 模型之前，需要先設置環境變量

```python
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
```

使用在線的 Hugging Face 模型

```python
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":64})
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
print(llm_chain.run(question))
```

將 Hugging Face 模型直接拉到本地使用

```python
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

model_id = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)
print(local_llm('What is the capital of France? '))


template = """Question: {question} Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=local_llm)
question = "What is the capital of England?"
print(llm_chain.run(question))
```

將模型拉到本地使用的好處：

* 訓練模型
* 可以使用本地的 GPU
* 有些模型無法在 Hugging Face 運行

### **通過自然語言執行SQL命令**

我們通過 `SQLDatabaseToolkit` 或者 `SQLDatabaseChain` 都可以實現執行SQL命令的操作

```python
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI

db = SQLDatabase.from_uri("sqlite:///../notebooks/Chinook.db")
toolkit = SQLDatabaseToolkit(db=db)

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True
)

agent_executor.run("Describe the playlisttrack table")
```

```python
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

db = SQLDatabase.from_uri("mysql+pymysql://root:root@127.0.0.1/chinook")
llm = OpenAI(temperature=0)

db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
db_chain.run("How many employees are there?")
```

這裡可以參考這兩篇文檔：

[https://python.langchain.com/en/latest/modules/agents/toolkits/examples/sql\_database.html](https://python.langchain.com/en/latest/modules/agents/toolkits/examples/sql\_database.html)

[https://python.langchain.com/en/latest/modules/chains/examples/sqlite.html](https://python.langchain.com/en/latest/modules/chains/examples/sqlite.html)

## 總結

所有的案例都基本已經結束了，希望大家能通過這篇文章的學習有所收穫。這篇文章只是對 LangChain 一個初級的講解，高級的功能希望大家繼續探索。

並且因為 LangChain 迭代極快，所以後面肯定會隨著AI繼續的發展，還會迭代出更好用的功能，所以我非常看好這個開源庫。

希望大家能結合 LangChain 開發出更有創意的產品，而不僅僅只搞一堆各種一鍵搭建chatgpt聊天客戶端的那種產品。

這篇標題後面加了個 `01` 是我希望這篇文章只是一個開始，後面如何出現了更好的技術我還是希望能繼續更新下去這個系列。

本文章的所有範例代碼都在這裡，祝大家學習愉快。

https://colab.research.google.com/drive/1ArRVMiS-YkhUlobHrU6BeS8fF57UeaPQ?usp=sharing 
