# content-moderation
#Detecting sensitive items like weapons in images to help moderate content for users.


in order to run this program fist start cmd as an admin (just to be safe)
then go to the files where you wll have the project and start an env we'll call it moderation
so for me I run the command:

cd  C:\Users\user\Downloads\content-moderation-main\content-moderation-main

python -m venv moderation


#if you already have a python environment activate it#
to do this from cmd type the path to the activation folder which should now be:

you-moderation-folder\what you named your environment\Scripts\Activate

so for me i would run:
C:\Users\user\Downloads\content-moderation-main\content-moderation-main\moderation\Scripts\Activate

#necessary packages#
you will now need to pip install the necessary packages:

pip install numpy
pip install opencv-python
pip install tensorflow
pip install scikit-learn
pip install imutils
pip install Pillow
pip install tkinterdnd2


#with the environment active and the necessary packages# 

#training#
	formatting the data
in order to format the data properly, create a folder which we called train, but you may call anything

Then inside that folder, create folders for each class including a random controll class with images that have nothing to do with the others
to avoid overfitting. For us, we created the folders data, knife, pistol, rifle.

data was the control dataset per say with mostly random images that had nothing to do with anything else.

all teh other ones were explicit in nature

but for example we had the project folder at
C:\Users\user\Downloads\content-moderation-main\content-moderation-main

and train data at 
C:\Users\user\Downloads\content-moderation-main\content-moderation-main\train

and that included labeled folders like:
C:\Users\user\Downloads\content-moderation-main\content-moderation-main\train\data
C:\Users\user\Downloads\content-moderation-main\content-moderation-main\train\knife
C:\Users\user\Downloads\content-moderation-main\content-moderation-main\train\rifle
C:\Users\user\Downloads\content-moderation-main\content-moderation-main\train\pistol

and each of those directories had imaages in them like:
C:\Users\user\Downloads\content-moderation-main\content-moderation-main\train\knife\knifeimage.jpg

now to run this, you will have to pick a name for the file of the model, we used all.h5, but it could be anything 
as long as it's a .h5 file

then to run it, from the cmd console in the environment with all the packages installed,in the directory of the project
where the train folder is your folder of datasets and in this folder, run the command

python model_train.py -d train -m all.h5

*note - you may need significant memory resources given teh size and number of your datasets etc.
*note -  the model will automautically do a 25-75 test/train split

then wait for this to be done

#using the program/gui#
if you chose a h5 filename other tan all.h5 on line 76 of image_moderatinon.py
    app = ImageClassifier(model_path="all.h5")
	
change this to that files name and path

also if you named the classes other than data, knife, pistol and rifle, change them on line 16 of image_moderatinon.py
        self.class_labels = ['non-explicit', 'knife', 'pistol', 'rifle']
you will also have to update the indices on line 50:
        explicit_indices = [1, 2, 3]
		
the classes should be in alphabetical order of the class names in the train folder you used.
so in our isntance, data is first (but we called it non--explicit here, knife is second, pistol and then rifle.

then in order to run the image_moderation.y and use the gui, simply run:
pytho image_moderation.py

then, just drag and drop images into the gui,and a score will pop up.
