#include<iostream>
#include<string>

using namespace std;


class Weapon{
protected:
	string weaponName;
	
public:
	explicit Weapon(){}
	
	Weapon(string N){
		weaponName = N;
	}
	
	string getWeaponName(){return weaponName;}
	
	void setWeaponName(string a){weaponName = a;}
	
	~Weapon(){}
};

class ExoticWeapon : public Weapon{
	string origin;
	
public:
	ExoticWeapon() : Weapon(){}
	
	string getOrigin(){return origin;}
	
	void setOrigin(string o){origin = o;}
};


class Samurai{
protected:
	string name, samuraiWeapon;
	int numberOfWins, numberOfInjuries, numberOfDuels, age;
	
public:
	explicit Samurai(){}
	
	Samurai(string N, int w, int i, int d, int a){
		name = N;
		numberOfWins = w;
		numberOfInjuries = i;
		numberOfDuels = d;
		age = a;
		samuraiWeapon = "no weapon";
	}
	
	string getName(){return name;}
	string getSamuraiWeapon(){return samuraiWeapon;}
	int getNumberOfWins(){return numberOfWins;}
	int getNumberOfInjuries(){return numberOfInjuries;}
	int getNumberOfDuels(){return numberOfDuels;}
	int getAge(){return age;}
	
	void setName(string a){name = a;}
	void setSamuraiWeapon(string f){samuraiWeapon = f;}
	void setNumberOfWins(int b){numberOfWins = b;}
	void setNumberOfInjuries(int c){numberOfInjuries = c;}
	void setNumberOfDuels(int d){numberOfDuels = d;}
	void setAge(int e){age = e;}
	
	void printSamuraiDescription(Samurai x){
		cout << "Samurai name:" << x.name << " Duels:" << x.numberOfDuels << " Wins:" << x.numberOfWins << " Injuries:" << x.numberOfInjuries << endl;
	}
	
	void pickWeapon(Weapon &a){
		(age < 18) ? samuraiWeapon = "Wooden Sword" : samuraiWeapon = a.getWeaponName();			
	}
	
	~Samurai(){}
};

class Ninja : public Samurai{
	int grade;
	
public:
	Ninja() : Samurai(){}
	
	int getGrade(){return grade;}
	
	void setGrade(int g){grade = g;}
	
	
	void pickExoticWeapon(ExoticWeapon &a){
		samuraiWeapon = a.getWeaponName();
	}
};


class NinjaSchool{
	Ninja ninjaClass[10];
	int numberOfStudents;
	
public:
	NinjaSchool(){
		Ninja *ninjaClass;
		ninjaClass = new Ninja[10];
		numberOfStudents = 0;
	}
	
	int getNumberOfStudents(){return numberOfStudents;}

	void setNumberOfStudents(int s){numberOfStudents = s;}
	
	NinjaSchool operator++(){
        numberOfStudents++;
        
        return *this;
    }
        
	NinjaSchool operator++(int notused){
		NinjaSchool temp = *this;
        numberOfStudents++;

        return temp;
	}
				
	NinjaSchool operator--(){
        numberOfStudents--;
        
        return *this;
    }
			
	NinjaSchool operator--(int notused){
    	NinjaSchool temp = *this;
        numberOfStudents--;

        return temp;
	}
	
	
	Ninja startTrainingNinja(Samurai &a){
		Ninja temp;
		
		temp.setName(a.getName());
		temp.setNumberOfDuels(a.getNumberOfDuels());
		temp.setNumberOfInjuries(a.getNumberOfInjuries());
		temp.setNumberOfWins(a.getNumberOfWins());
		temp.setSamuraiWeapon(a.getSamuraiWeapon());
		temp.setAge(a.getAge());
		temp.setGrade(0);
		
		delete &a;
		
		return temp;
	}
	
	void addNinjaToNinjaClass(Ninja &a){
		if(numberOfStudents <= 9){
			if(a.getAge() >= 18){
			
				ninjaClass[numberOfStudents] = a;
				++numberOfStudents;
			
			}else cout << "This class is for grownups!" << endl;			
		}
	}
	
	string brawlWithNinjas(){
		int temp = 0;
		string name;
		
		for(int i = 0 ; i <= (numberOfStudents -1) ; i++) {
			if((ninjaClass[i].getGrade()) >= temp){
				temp = ninjaClass[i].getGrade();
				name = ninjaClass[i].getName();
			}
		}
		
		return name;
	}
};


void duel(Samurai &a, Samurai &b, bool winner){
	a.setNumberOfDuels(a.getNumberOfDuels() + 1);
	b.setNumberOfDuels(b.getNumberOfDuels() + 1);
	
	if(winner){
		a.setNumberOfWins(a.getNumberOfWins() + 1);
		b.setNumberOfInjuries(b.getNumberOfInjuries() + 1);
		
		if(a.getNumberOfWins() > 10) cout << "Samurai " << a.getName() << " is duelist." << endl;
	}else{
		b.setNumberOfWins(b.getNumberOfWins() + 1);
		a.setNumberOfInjuries(a.getNumberOfInjuries() + 1);
		
		if(b.getNumberOfWins() > 10) cout << "Samurai " << b.getName() << " is duelist." << endl;
	}
}

string duelForYoungSamurais(Samurai &a, Samurai &b){
	if(a.getAge() > 18 || b.getAge() > 18){
		
		return "Invalid duel!";
		
	}else if((a.getSamuraiWeapon() != "Wooden Sword") || (b.getSamuraiWeapon() != "Wooden Sword")){
		
		return "Duel postponed!";
		
	}else{
		
		return a.getName() + " duels " + b.getName() + "!";
		
	}
}

string duelWithWeapons(Samurai &a, Samurai &b){
	if(a.getAge() < 18 || b.getAge() < 18){
		
		return "This duel is for adults only!";
				
	}else if((a.getSamuraiWeapon() != "Rock" && a.getSamuraiWeapon() != "Scissors" && a.getSamuraiWeapon() != "Paper") || (b.getSamuraiWeapon() != "Rock" && b.getSamuraiWeapon() != "Scissors" && b.getSamuraiWeapon() != "Paper")){
		
		return "Strange Duel!";
		
	}else if(a.getSamuraiWeapon() == b.getSamuraiWeapon()){
		
		return "The duel is draw!";
		
	}else{
		if(a.getSamuraiWeapon() == "Rock"){
			
			if(b.getSamuraiWeapon() == "Scissors") return a.getName();
			else return b.getName();
			
		}else if(a.getSamuraiWeapon() == "Scissors"){
			
			if(b.getSamuraiWeapon() == "Paper") return a.getName();
			else return b.getName();			
			
		}else{
			
			if(b.getSamuraiWeapon() == "Rock") return a.getName();
			else return b.getName();
			
		}
	}
}


//Ninja NinjaSchool :: getStudent(int d){return ninjaClass[d];} //don't forget to declare it in the class;

int main(){
	{ //testing exotic weapons
	Ninja n1;
	ExoticWeapon e1;
	e1.setWeaponName("Cringebringer");
	
	n1.pickExoticWeapon(e1);
	
	cout << "The Ninja has picked the weapon " << n1.getSamuraiWeapon() << endl << endl;
	}
	
	{ //testing opperators	
	NinjaSchool C1;
	
	C1.setNumberOfStudents(7);
	
	cout << "The initial number of students is " << C1.getNumberOfStudents() << endl;
	
	C1++;
	++C1;
	
	cout << "The number of students is now " << C1.getNumberOfStudents() << endl;
	
	C1--;
	--C1;
	C1--;
	
	cout << "The number of students is now " << C1.getNumberOfStudents() << endl << endl;
	}
	
	{ //test Ninja training
	Samurai s1("Ilias", 99, 1, 100, 21);
	NinjaSchool C1;
	Ninja N1;
	
	cout << "Starting to train Samurai " << s1.getName() << " as a Ninja" << endl;
	N1 = C1.startTrainingNinja(s1);
	
	cout << "Samurai " << N1.getName() << ", " << N1.getAge() << " years old, has been successfuly retrained as a Ninja of grade " << N1.getGrade() << endl;
	cout << "His stats are: " << N1.getNumberOfDuels() << " Duels, " << N1.getNumberOfWins() << " Wins" << endl << endl;
	}
	
	{ //testing add ninja to class
	NinjaSchool C1;
		
	cout << "Creating the samurais and training them..." << endl << "." << endl << "."<< endl << "." << endl;
	Samurai s1("Ilias", 99, 1, 100, 19), s2("George", 39, 1, 40, 18), s3("Panos", 19, 1, 20, 16);
	Ninja N1, N2, N3;
	N1 = C1.startTrainingNinja(s1);
	N2 = C1.startTrainingNinja(s2);
	N3 = C1.startTrainingNinja(s3);
	cout << "Samurais " << N1.getName() << ", " << N2.getName() << " and " << N3.getName() << " have been successfuly retrained." << endl;
	
	cout << "Adding the samurais to their class: " << endl << endl;
	C1.addNinjaToNinjaClass(N1);
	C1.addNinjaToNinjaClass(N2);
	C1.addNinjaToNinjaClass(N3);
	
	cout << "The members of the class are: " << endl;
	for(int i = 0 ; i <= (C1.getNumberOfStudents() - 1) ; i++){
		
		Ninja N = C1.getStudent(i);
		cout << N.getName() << ": Grade " << N.getGrade() << ", Wins: " << N.getNumberOfWins() << endl << endl << endl;
		
	}  
	}

	{ //test brawl
	NinjaSchool C1;
	Samurai s1("Ilias", 99, 1, 100, 19), s2("George", 39, 1, 40, 18), s3("Panos", 19, 1, 20, 123);
	Samurai s4("Natasa", 99, 1, 100, 22), s5("Emilia", 39, 1, 40, 25), s6("Bron", 19, 1, 20, 32);
	Ninja N1, N2, N3, N4, N5, N6;
	N1 = C1.startTrainingNinja(s1);
	N2 = C1.startTrainingNinja(s2);
	N3 = C1.startTrainingNinja(s3);
	N4 = C1.startTrainingNinja(s4);
	N5 = C1.startTrainingNinja(s5);
	N6 = C1.startTrainingNinja(s6);
	
	N1.setGrade(400);
	N2.setGrade(9);
	N3.setGrade(400);
	N4.setGrade(85);
	N5.setGrade(6);
	N6.setGrade(3);
	
	C1.addNinjaToNinjaClass(N1);
	C1.addNinjaToNinjaClass(N2);
	C1.addNinjaToNinjaClass(N3);
	C1.addNinjaToNinjaClass(N4);
	C1.addNinjaToNinjaClass(N5);
	C1.addNinjaToNinjaClass(N6);
	
	cout << "The Ninja class has " << C1.getNumberOfStudents() << endl;
	cout <<"The winner of the brawl is " << C1.brawlWithNinjas() << endl;
	}

	{ //test adding more than 10 Ninjas to Class
		// (Requires a cout << "."; at the end of add (inside the if))
	NinjaSchool C1;	
	Samurai s1("Ilias", 99, 1, 100, 19), s2("George", 39, 1, 40, 18), s3("Panos", 19, 1, 20, 123);
	Samurai s4("Natasa", 99, 1, 100, 22), s5("Emilia", 39, 1, 40, 25), s6("Bron", 19, 1, 20, 32);
	Samurai s7("Katrina", 99, 1, 100, 22), s8("Frank", 39, 1, 40, 25), s9("Tommy", 19, 1, 20, 32);
	Samurai s10("Sandy", 99, 1, 100, 22), s11("Moromo", 39, 1, 40, 25), s12("Mark", 19, 1, 20, 32);
	Ninja N1 = C1.startTrainingNinja(s1), N2 = C1.startTrainingNinja(s2), N3 = C1.startTrainingNinja(s3);
	Ninja N4 = C1.startTrainingNinja(s4), N5 = C1.startTrainingNinja(s5), N6 = C1.startTrainingNinja(s6);
	Ninja N7 = C1.startTrainingNinja(s7), N8 = C1.startTrainingNinja(s8), N9 = C1.startTrainingNinja(s9);
	Ninja N10 = C1.startTrainingNinja(s10), N11 = C1.startTrainingNinja(s11), N12 = C1.startTrainingNinja(s12);
	
	C1.addNinjaToNinjaClass(N1);
	C1.addNinjaToNinjaClass(N2);
	C1.addNinjaToNinjaClass(N3);
	C1.addNinjaToNinjaClass(N4);
	C1.addNinjaToNinjaClass(N5);
	C1.addNinjaToNinjaClass(N6);
	C1.addNinjaToNinjaClass(N7);
	C1.addNinjaToNinjaClass(N8);
	C1.addNinjaToNinjaClass(N9);
	C1.addNinjaToNinjaClass(N10);
	C1.addNinjaToNinjaClass(N11);
	C1.addNinjaToNinjaClass(N12);
	}

	return 0;
}