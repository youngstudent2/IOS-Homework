//
//  ViewController.swift
//  TAP Light
//
//  Created by youngstudent2 on 2020/9/21.
//  Copyright © 2020 youngstudent2. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }


    @IBOutlet weak var myLabel: UILabel!
    @IBOutlet weak var myView: UIView!
    
    @IBAction func changeLight(){
        if(myView.backgroundColor == UIColor.white){
            myView.backgroundColor = UIColor.black
            myLabel.text = "OFF"
   
        }
        else{
            myView.backgroundColor = UIColor.white
            myLabel.text = "ON"

        }
        
    }
    
}

