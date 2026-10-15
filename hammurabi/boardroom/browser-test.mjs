import { chromium } from 'playwright';
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
try{
 const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8765',{waitUntil:'domcontentloaded',timeout:20000});
 await page.waitForFunction(()=>document.querySelectorAll('.person').length===25,{},{timeout:15000});
 await page.locator('#all').click();if(await page.locator('.tile').count()!==25)throw Error('Expected full board');
 await page.locator('#clear').click();if(await page.locator('.tile').count()!==0)throw Error('Expected empty board');
 await page.getByRole('button',{name:/Atlas · Chief Executive/}).click();
 await page.locator('#message').fill('Hello Atlas');await page.locator('#send').click();
 await page.getByText(/Local model not connected. Install/).waitFor();
 if(errors.length)throw Error(errors.join('\n'));
 console.log('PASS: 25 seats, full-board selection, clear, individual selection, honest missing-model handling; no page errors');
}finally{await browser.close();}
