from Tkinter import * 
from PIL import Image, ImageTk
import os
shirt_color_dict = {"DISABLED.png":'-1',"full_shirt_red.png":'0',"full_shirt_creme.png":'1',"full_shirt_yellow.png":'2',"full_shirt_blue.png":'3',"full_shirt_pink.png":'4'}
pant_color_dict = {"DISABLED.png":'-1',"fullpant_olive.png":'0',"fullpant_white.png":'1',"fullpant_blue.png":'2',"fullpant_green.png":'3',"fullpant_red.png":'4',"fullpant_brown.png":'5'}
glass_color_dict = {"DISABLED.png":'-1',"glasses0.png":'0',"glasses1.png":'1',"glasses2.png":'2',"glasses3.png":'3'}

def clicked():
    shirtcolor_val = shirt_color_dict[str(shirtcolor.get())]
    pantcolor_val = pant_color_dict[str(pantcolor.get())]
    glasssrc = glass_color_dict[str(glasscolor.get())]
    shirt_size = str(shirtsize.get())
    pant_size = str(pantsize.get())
    print(shirtcolor_val,pantcolor_val,glasssrc,shirt_size,pant_size)
    os.system("python Main.py "+shirtcolor_val+" "+pantcolor_val+" "+glasssrc+" "+shirt_size+" "+pant_size)

def clicked1():
    print("Running pose detector")
    os.system("python Pose_est.py")

def changeshirt():

    load = Image.open("images/"+str(shirtcolor.get()))
    if str(shirtcolor.get()) == "DISABLED.png":
            load = load.resize((200, 200), Image.ANTIALIAS)
    else:
        load = load.resize((150, 200), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    shirt.configure(image=render)
    shirt.image = render
    # if str(shirtcolor.get()) == "DISABLED.png":
    #     shirt.grid(column=0,row=1,padx=0)
    # else:
    #     shirt.grid(column=0,row=1,padx=100)

def changepant():

    load = Image.open("images/"+str(pantcolor.get()))
    if str(pantcolor.get()) == "DISABLED.png":
            load = load.resize((200, 200), Image.ANTIALIAS)
    else:
        load = load.resize((100, 200), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    pant.configure(image=render)
    pant.image = render

def changeglass():
    
    load = Image.open("images/"+str(glasscolor.get()))
    if str(glasscolor.get()) == "DISABLED.png":
            load = load.resize((200, 200), Image.ANTIALIAS)
    else:
        load = load.resize((200, 100), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    glass.configure(image=render)
    glass.image = render
    

window = Tk()  
window.title("Virtual Dressing Room")
window.geometry('1000x800')
window.resizable(0,0)
window.iconbitmap("images/vd.ico")
window.configure(background="light blue")
#Entering the event main loop  
# txt = Entry(window,width=10)
# txt.grid(column=1, row=0)
# txt.focus()
lbl = Label(window, text="Virtual Dresser", font=("Comic Sans MS", 50),background="light blue")
lbl.grid(column=1, row=0)
load = Image.open("images/DISABLED.png")
load = load.resize((200, 200), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
shirt = Label(window, image=render,background="light blue")
shirt.image = render
shirt.grid(column=0,row=1,padx=0)
load = Image.open("images/DISABLED.png")
load = load.resize((200, 200), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
pant = Label(window, image=render,background="light blue")
pant.image = render
pant.grid(column=1,row=1)
load = Image.open("images/DISABLED.png")
load = load.resize((200, 200), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
glass = Label(window, image=render,background="light blue")
glass.image = render
glass.grid(column=2,row=1)
btn = Button(window, text="Launch Trial", bg="orange", fg="red", font=("Comic Sans MS", 15),width=25,command=clicked)
btn.grid(column=1, row=10,pady=20)
btn1 = Button(window, text="View Pose Estimation", bg="orange", fg="dark green", font=("Comic Sans MS", 10),width=15,command=clicked1)
btn1.grid(column=2, row=10,pady=20)
shirtcolor = StringVar()
pantcolor = StringVar()
glasscolor = StringVar()
shirtsize = StringVar()
pantsize = StringVar()


shirt_rad1 = Radiobutton(window,variable=shirtcolor,width=10, indicator = 0, background = "light blue",text='Disable', value="DISABLED.png",command=changeshirt,justify = LEFT,compound=LEFT, font=("Comic Sans MS", 10))
shirt_rad1.select()
shirt_rad2 = Radiobutton(window,variable=shirtcolor,width=10, indicator = 0, background = "light blue",text='RED', value="full_shirt_red.png",command=changeshirt,justify = LEFT,compound=LEFT, font=("Comic Sans MS", 10))
shirt_rad3 = Radiobutton(window,variable=shirtcolor,width=10, indicator = 0, background = "light blue",text='CREME', value="full_shirt_creme.png",command=changeshirt,justify = LEFT,compound=LEFT, font=("Comic Sans MS", 10))
shirt_rad4 = Radiobutton(window,variable=shirtcolor,width=10, indicator = 0, background = "light blue",text='YELLOW', value="full_shirt_yellow.png",command=changeshirt,justify = LEFT,compound=LEFT, font=("Comic Sans MS", 10))
shirt_rad5 = Radiobutton(window,variable=shirtcolor,width=10, indicator = 0, background = "light blue",text='BLUE', value="full_shirt_blue.png",command=changeshirt,justify = LEFT,compound=LEFT, font=("Comic Sans MS", 10))
shirt_rad6 = Radiobutton(window,variable=shirtcolor,width=10, indicator = 0, background = "light blue",text='PINK', value="full_shirt_pink.png",command=changeshirt,justify = LEFT,compound=LEFT, font=("Comic Sans MS", 10))
shirt_rad1.grid(column=0, row=2,sticky='w',padx=100)
shirt_rad2.grid(column=0, row=3,sticky='w',padx=100)
shirt_rad3.grid(column=0, row=4,sticky='w',padx=100)
shirt_rad4.grid(column=0, row=5,sticky='w',padx=100)
shirt_rad5.grid(column=0, row=6,sticky='w',padx=100)
shirt_rad6.grid(column=0, row=7,sticky='w',padx=100)

pant_rad1 = Radiobutton(window,variable=pantcolor,width=10, indicator = 0, background = "light blue",text='Disable', value="DISABLED.png",command=changepant,justify = LEFT, font=("Comic Sans MS", 10))
pant_rad1.select()
pant_rad2 = Radiobutton(window,variable=pantcolor,width=10, indicator = 0, background = "light blue",text='OLIVE', value="fullpant_olive.png",command=changepant,justify = LEFT, font=("Comic Sans MS", 10))
pant_rad3 = Radiobutton(window,variable=pantcolor,width=10, indicator = 0, background = "light blue",text='WHITE', value="fullpant_white.png",command=changepant,justify = LEFT, font=("Comic Sans MS", 10))
pant_rad4 = Radiobutton(window,variable=pantcolor,width=10, indicator = 0, background = "light blue",text='BLUE',value="fullpant_blue.png",command=changepant,justify = LEFT, font=("Comic Sans MS", 10))
pant_rad5 = Radiobutton(window,variable=pantcolor,width=10, indicator = 0, background = "light blue",text='GREEN', value="fullpant_green.png",command=changepant,justify = LEFT, font=("Comic Sans MS", 10))
pant_rad6 = Radiobutton(window,variable=pantcolor,width=10, indicator = 0, background = "light blue",text='RED', value="fullpant_red.png",command=changepant,justify = LEFT, font=("Comic Sans MS", 10))
pant_rad7 = Radiobutton(window,variable=pantcolor,width=10, indicator = 0, background = "light blue",text='BROWN', value="fullpant_brown.png",command=changepant,justify = LEFT, font=("Comic Sans MS", 10))
pant_rad1.grid(column=1, row=2,sticky='w',padx=200)
pant_rad2.grid(column=1, row=3,sticky='w',padx=200)
pant_rad3.grid(column=1, row=4,sticky='w',padx=200)
pant_rad4.grid(column=1, row=5,sticky='w',padx=200)
pant_rad5.grid(column=1, row=6,sticky='w',padx=200)
pant_rad6.grid(column=1, row=7,sticky='w',padx=200)
pant_rad7.grid(column=1, row=8,sticky='w',padx=200)

glass_rad1 = Radiobutton(window,variable=glasscolor,width=10, indicator = 0, background = "light blue",text='Disable', value="DISABLED.png",command=changeglass,justify = LEFT, font=("Comic Sans MS", 10))
glass_rad1.select()
glass_rad2 = Radiobutton(window,variable=glasscolor,width=10, indicator = 0, background = "light blue",text='GREY', value="glasses0.png",command=changeglass,justify = LEFT, font=("Comic Sans MS", 10))
glass_rad3 = Radiobutton(window,variable=glasscolor,width=10, indicator = 0, background = "light blue",text='GOLD-RIM', value="glasses1.png",command=changeglass,justify = LEFT, font=("Comic Sans MS", 10))
glass_rad4 = Radiobutton(window,variable=glasscolor,width=10, indicator = 0, background = "light blue",text='RED',value="glasses2.png",command=changeglass,justify = LEFT, font=("Comic Sans MS", 10))
glass_rad5 = Radiobutton(window,variable=glasscolor,width=10, indicator = 0, background = "light blue",text='THUG-LIFE', value="glasses3.png",command=changeglass,justify = LEFT, font=("Comic Sans MS", 10))
glass_rad1.grid(column=2, row=2,sticky='w',padx=50)
glass_rad2.grid(column=2, row=3,sticky='w',padx=50)
glass_rad3.grid(column=2, row=4,sticky='w',padx=50)
glass_rad4.grid(column=2, row=5,sticky='w',padx=50)
glass_rad5.grid(column=2, row=6,sticky='w',padx=50)

shirt_sizer = Frame(window, background = "light blue")
shirt_sizer.grid(column=0,row=9,sticky='w',padx=50,pady=10)
for i,text in enumerate(['XS','S','M','L','XL']):
    Label(shirt_sizer, text = text, background = "light blue").grid(row=0,column=i,padx=10,sticky="w")

slider1 = Scale(shirt_sizer,showvalue=0,from_=1,to=5, length=180,orient=HORIZONTAL,relief="sunken", background = "light blue",variable=shirtsize)
slider1.grid(row = 1, column = 0, columnspan = 5,ipadx=0)

pant_sizer = Frame(window, background = "light blue")
pant_sizer.grid(column=1,row=9,sticky='w',padx=150,pady=10)
for i,text in enumerate(['XS','S','M','L','XL']):
    Label(pant_sizer, text = text, background = "light blue").grid(row=0,column=i,padx=10,sticky="w")

slider2 = Scale(pant_sizer,showvalue=0,from_=1,to=5, length=180,orient=HORIZONTAL,relief="sunken", background = "light blue",variable=pantsize)
slider2.grid(row = 1, column = 0, columnspan = 5,ipadx=0)
lbl2 = Label(window, text="Click to start trial\n Press Q anytime to quit", font=("Comic Sans MS", 10),background="light blue")
lbl2.grid(row = 11,column = 1,pady=10)

window.mainloop() 