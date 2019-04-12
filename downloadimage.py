def downloadImage(searchTerm):
    cxid = "017848639769224862235:b7gsieimvak"
    apikey = "AIzaSyB5Jv4i8B1uhGmUsSQ-k8LIRraN2pR9rUI"

    #GET CXID AND API KEY FROM : https://developers.google.com/custom-search/v1/overview

    from googleapiclient.discovery import build
    import json
    import requests
    #from pprint import pprint
    from PIL import Image
    #use google custom search to retrieve the first image the serach returns
    service = build("customsearch", "v1",
            developerKey=apikey) #to use google custom search engine

    res = service.cse().list(
      q=searchTerm, #query to be searched
      cx=cxid, #custom search engine ID
      num=10, #no of images you want to download
      searchType='image', #type of file
      fileType='jpg', #extension
      safe='off', #not child friendly
    ).execute() #run this engine 

    #pprint(res) #print JSON if you want to see the structure
    linkImg=""
    
     #extract link from returned JSON String and download image
    i=40
    for data in res['items']:
        linkImg=data['link']

        f=open('test2/image'+str(i)+'.jpeg','wb') 
        f.write(requests.get(linkImg).content)
        f.close()

        print("Downloaded from :"+ linkImg)
        i+=1

query=input("Topic : ")
downloadImage(query)