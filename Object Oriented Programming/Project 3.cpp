#include<iostream>
#include<string>

using namespace std;


class Weapon{
	string weaponName;
	
public:
	Weapon(string N){
		weaponName = N;
		
		cout << "Weapon created!" << endl;
	}
	
	string getWeaponName(){return weaponName;}
	
	void setWeaponName(string a){weaponName = a;}
	
	~Weapon(){
		cout << "Weapon destroyed!" << endl;
	}
	
};

class Samurai{
	string name, samuraiWeapon;
	int numberOfWins, numberOfInjuries, numberOfDuels, age;
	
public:
	Samurai(string N, int w, int i, int d, int a){
		name = N;
		numberOfWins = w;
		numberOfInjuries = i;
		numberOfDuels = d;
		age = a;
		samuraiWeapon = "no weapon";
		
		cout << "Samurai ready for duel." << endl;
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
	
	~Samurai(){
		cout << "Samurai " << name << " deleted." << endl;
	}
	
};


void duel(Samurai &a, Samurai &b, bool winner){
	a.setNumberOfDuels(a.getNumberOfDuels() + 1);
	b.setNumberOfDuels(b.getNumberOfDuels() + 1);
	
	if(winner){
		a.setNumberOfWins(a.getNumberOfWins() + 1);
		b.setNumberOfInjuries(b.getNumberOfInjuries() + 1);
		
		if(a.getNumberOfWins() > 10) cout << "Samurai " << a.getName() << " is duelist.";
	}else{
		b.setNumberOfWins(b.getNumberOfWins() + 1);
		a.setNumberOfInjuries(a.getNumberOfInjuries() + 1);
		
		if(b.getNumberOfWins() > 10) cout << "Samurai " << b.getName() << " is duelist.";
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


int main(){
	Weapon w1("Rock");
	Weapon w2("Paper");
	Weapon w3("Scissors");
	Weapon w4("Pencil");
	
	Samurai S1("Sofoklis", 11, 5, 16, 19);
	Samurai S2("Zafiris", 3, 7, 10, 19);
	S1.pickWeapon(w3);
	cout << endl << "Samurai " << S1.getName() << " has picked the weapon " << S1.getSamuraiWeapon() << "." << endl;
	S2.pickWeapon(w2);
	cout << "Samurai " << S2.getName() << " has picked the weapon " << S2.getSamuraiWeapon() << "." << endl;
	cout << duelForYoungSamurais(S1,S2) << endl;	
	cout << duelWithWeapons(S2,S1) << endl;
	

	cout << "Creating the Duel Weapons:" << endl << endl;

	Weapon w1("Rock");
	cout << endl << "The name of the new weapon is: " << w1.getWeaponName() << endl;
	Weapon w2("Paper");
	cout << endl << "The name of the new weapon is: " << w2.getWeaponName() << endl;
	Weapon w3("Scissors");
	cout << endl << "The name of the new weapon is: " << w3.getWeaponName() << endl << endl << endl;

	
	cout << "Duel for young Samurais about to begin:" << endl << endl << endl;
	
	cout << "Summoning the Samurais:" << endl << endl;
	Samurai S1("Sofoklis", 11, 5, 16, 17);
	cout << endl << "Samurai " << S1.getName() << "' weapon is: " << S1.getSamuraiWeapon() << endl;
	Samurai S2("Zafiris", 3, 7, 10, 16);
	cout << endl << "Samurai " << S2.getName() << "' weapon is: " << S2.getSamuraiWeapon() << endl << endl;
	
	cout << "Samurais picking their weapons:" << endl << endl;
	S1.pickWeapon(w1);
	cout << "Samurai " << S1.getName() << " has picked the weapon " << S1.getSamuraiWeapon() << "." << endl;
	cout << "This happened because " << S1.getName() << " is " << S1.getAge() << " years old." << endl;
	S2.pickWeapon(w2);
	cout << "Samurai " << S2.getName() << " has picked the weapon " << S2.getSamuraiWeapon() << "." << endl;
	cout << "This happened because " << S2.getName() << " is " << S2.getAge() << " years old." << endl << endl;	
	
	cout << duelForYoungSamurais(S1,S2) << endl;
	cout << "Samurai " << S2.getName() << " died!" << endl << endl << endl;
	
	
	system("pause");
	system("cls");
	
	S1.setAge(S1.getAge() + 3);
	cout << "Three years passed!" << endl << "Samurai " << S1.getName() << "'s age is now " << S1.getAge() << " years old." << endl << endl << endl;
	cout << "Beware! The most powerful samurai arrives!" << endl;
	Samurai S("Ilias", 0, 0, 1000, 19);
	cout << endl << "Samurai " << S.getName() << "' weapon is: " << S.getSamuraiWeapon() << endl << endl;
	
	cout << "He is challenging " << S1.getName() << " to a fight!" << endl;
	cout << "Available weapons: " << endl << w1.getWeaponName() << endl << w2.getWeaponName() << endl << w3.getWeaponName() << endl << endl << endl;
	cout << "The duel for adult Samurais is about to begin." << endl << endl;
	
	S.pickWeapon(w1);
	S1.pickWeapon(w3);
	
	cout << "Samurai " << S.getName() << " picked the weapon " << S.getSamuraiWeapon() << "." << endl;
	cout << "Samurai " << S1.getName() << " picked the weapon " << S1.getSamuraiWeapon() << "." << endl << endl;
	
	cout << "Duel is starting." << endl << endl << endl;
	cout << "The winner is " << duelWithWeapons(S,S1) << "!" << endl << endl << "Samurai " << S1.getName() << " died." << endl << endl;
	
	return 0;
}