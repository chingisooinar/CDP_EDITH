from urllib.request import urlopen
from bs4 import BeautifulSoup as bs


idx = 1
def crawlFromUrl(keyword,url, base):
    global idx
    
    html = urlopen(url)  
    bsObject = bs(html, "html.parser") 
    
    try:
        for thumb in bsObject.find_all(class_="preview"):
            src = thumb["src"]
            html = urlopen(url)  
            thumb_split = thumb["src"].split("?")
            id = thumb_split[-1]
            src = base + "&s=view&id=" + id
            
            eachHtml = urlopen(src)
            originObject = bs(eachHtml, "html.parser")
            
            content = originObject.find(class_="content")
            originSrc = content.find('img')['src']
            
            with urlopen(originSrc) as s:
                with open("./safebooru/"+keyword+"/safebooru_"+keyword+"_" + str(idx)+'.jpg', 'wb') as b:
                    data = s.read()
                    b.write(data)
            print("downloaded {} pictures".format(idx))
            idx += 1
            
    except Exception as ex:
        print(ex)
        

if __name__ == '__main__':
   
    base = 'https://safebooru.org/index.php?page=post'
    
    keyword = input("type keyword : ")    
    pid = 0
    for itr in range(2000):
        url = base
        url += '&s=list'
        url += '&tags='
        url += keyword
        pid = itr * 40
        url += '&pid='
        url += str(pid)
        print(url)
        crawlFromUrl(keyword, url, base)


print('done')
