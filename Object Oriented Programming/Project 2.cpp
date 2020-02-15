#include<iostream>
#include<string>

using namespace std;

class Samurai{
	string name;
	int numberOfWins, numberOfInjuries, numberOfDuels;
	
public:
	Samurai(string N, int w, int i, int d){
		name = N;
		numberOfWins = w;
		numberOfInjuries = i;
		numberOfDuels = d;
		
		cout << "Samurai ready for duel." << endl;	
	};
	
	string getName(){return name;}
	int getNumberOfWins(){return numberOfWins;}
	int getNumberOfInjuries(){return numberOfInjuries;}
	int getNumberOfDuels(){return numberOfDuels;}
	
	void setName(string a){name = a;}
	void setnumberOfWins(int b){numberOfWins = b;}
	void setnumberOfInjuries(int c){numberOfInjuries = c;}
	void setnumberOfDuels(int d){numberOfDuels = d;}
	
	void printSamuraiDescription(Samurai x){
		cout << "Samurai name:" << x.name << " Duels:" << x.numberOfDuels << " Wins:" << x.numberOfWins << " Injuries:" << x.numberOfInjuries << endl;
	}
	
	~Samurai(){
		cout << "Samurai " << name << " deleted." << endl;
	};
};

void duel(Samurai &a, Samurai &b, bool winner){
	a.setnumberOfDuels(a.getNumberOfDuels() + 1);
	b.setnumberOfDuels(b.getNumberOfDuels() + 1);
	
	if(winner){
		
		a.setnumberOfWins(a.getNumberOfWins() + 1);
		b.setnumberOfInjuries(b.getNumberOfInjuries() + 1);
		
		if(a.getNumberOfWins() > 10) cout << "Samurai " << a.getName() << " is duelist." << endl;
		
	}else{
		
		b.setnumberOfWins(b.getNumberOfWins() + 1);
		a.setnumberOfInjuries(a.getNumberOfInjuries() + 1);
		
		if(b.getNumberOfWins() > 10) cout << "Samurai " << b.getName() << " is duelist." << endl;
		
	}
}


int main(){
	Samurai a("Ilias", 10, 5, 15);
	cout << endl;
	Samurai b("Zafeiris", 3, 5, 8); 
	cout << endl;
	
	cout << "Before the duel:" << endl;
	a.printSamuraiDescription(a);
	cout << endl;
	b.printSamuraiDescription(b);
	cout << endl;
	
	duel(a, b, 1);
	cout << endl;
	
	cout << "After the duel:" << endl;
	a.printSamuraiDescription(a);
	cout << endl;
	b.printSamuraiDescription(b);
	cout << endl;
}
