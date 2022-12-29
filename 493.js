/*! For license information please see 493.js.LICENSE.txt */
!function(t,e){if("object"==typeof exports&&"object"==typeof module)module.exports=e();else if("function"==typeof define&&define.amd)define([],e);else{var r=e();for(var n in r)("object"==typeof exports?exports:t)[n]=r[n]}}(self,(function(){return function(){var t,e,r={493:function(t,e,r){function n(t){return n="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(t){return typeof t}:function(t){return t&&"function"==typeof Symbol&&t.constructor===Symbol&&t!==Symbol.prototype?"symbol":typeof t},n(t)}function o(){"use strict";o=function(){return t};var t={},e=Object.prototype,r=e.hasOwnProperty,i=Object.defineProperty||function(t,e,r){t[e]=r.value},a="function"==typeof Symbol?Symbol:{},c=a.iterator||"@@iterator",u=a.asyncIterator||"@@asyncIterator",s=a.toStringTag||"@@toStringTag";function f(t,e,r){return Object.defineProperty(t,e,{value:r,enumerable:!0,configurable:!0,writable:!0}),t[e]}try{f({},"")}catch(t){f=function(t,e,r){return t[e]=r}}function l(t,e,r,n){var o=e&&e.prototype instanceof y?e:y,a=Object.create(o.prototype),c=new P(n||[]);return i(a,"_invoke",{value:S(t,r,c)}),a}function h(t,e,r){try{return{type:"normal",arg:t.call(e,r)}}catch(t){return{type:"throw",arg:t}}}t.wrap=l;var p={};function y(){}function d(){}function v(){}var g={};f(g,c,(function(){return this}));var m=Object.getPrototypeOf,b=m&&m(m(k([])));b&&b!==e&&r.call(b,c)&&(g=b);var w=v.prototype=y.prototype=Object.create(g);function x(t){["next","throw","return"].forEach((function(e){f(t,e,(function(t){return this._invoke(e,t)}))}))}function O(t,e){function o(i,a,c,u){var s=h(t[i],t,a);if("throw"!==s.type){var f=s.arg,l=f.value;return l&&"object"==n(l)&&r.call(l,"__await")?e.resolve(l.__await).then((function(t){o("next",t,c,u)}),(function(t){o("throw",t,c,u)})):e.resolve(l).then((function(t){f.value=t,c(f)}),(function(t){return o("throw",t,c,u)}))}u(s.arg)}var a;i(this,"_invoke",{value:function(t,r){function n(){return new e((function(e,n){o(t,r,e,n)}))}return a=a?a.then(n,n):n()}})}function S(t,e,r){var n="suspendedStart";return function(o,i){if("executing"===n)throw new Error("Generator is already running");if("completed"===n){if("throw"===o)throw i;return{value:void 0,done:!0}}for(r.method=o,r.arg=i;;){var a=r.delegate;if(a){var c=L(a,r);if(c){if(c===p)continue;return c}}if("next"===r.method)r.sent=r._sent=r.arg;else if("throw"===r.method){if("suspendedStart"===n)throw n="completed",r.arg;r.dispatchException(r.arg)}else"return"===r.method&&r.abrupt("return",r.arg);n="executing";var u=h(t,e,r);if("normal"===u.type){if(n=r.done?"completed":"suspendedYield",u.arg===p)continue;return{value:u.arg,done:r.done}}"throw"===u.type&&(n="completed",r.method="throw",r.arg=u.arg)}}}function L(t,e){var r=e.method,n=t.iterator[r];if(void 0===n)return e.delegate=null,"throw"===r&&t.iterator.return&&(e.method="return",e.arg=void 0,L(t,e),"throw"===e.method)||"return"!==r&&(e.method="throw",e.arg=new TypeError("The iterator does not provide a '"+r+"' method")),p;var o=h(n,t.iterator,e.arg);if("throw"===o.type)return e.method="throw",e.arg=o.arg,e.delegate=null,p;var i=o.arg;return i?i.done?(e[t.resultName]=i.value,e.next=t.nextLoc,"return"!==e.method&&(e.method="next",e.arg=void 0),e.delegate=null,p):i:(e.method="throw",e.arg=new TypeError("iterator result is not an object"),e.delegate=null,p)}function j(t){var e={tryLoc:t[0]};1 in t&&(e.catchLoc=t[1]),2 in t&&(e.finallyLoc=t[2],e.afterLoc=t[3]),this.tryEntries.push(e)}function E(t){var e=t.completion||{};e.type="normal",delete e.arg,t.completion=e}function P(t){this.tryEntries=[{tryLoc:"root"}],t.forEach(j,this),this.reset(!0)}function k(t){if(t){var e=t[c];if(e)return e.call(t);if("function"==typeof t.next)return t;if(!isNaN(t.length)){var n=-1,o=function e(){for(;++n<t.length;)if(r.call(t,n))return e.value=t[n],e.done=!1,e;return e.value=void 0,e.done=!0,e};return o.next=o}}return{next:_}}function _(){return{value:void 0,done:!0}}return d.prototype=v,i(w,"constructor",{value:v,configurable:!0}),i(v,"constructor",{value:d,configurable:!0}),d.displayName=f(v,s,"GeneratorFunction"),t.isGeneratorFunction=function(t){var e="function"==typeof t&&t.constructor;return!!e&&(e===d||"GeneratorFunction"===(e.displayName||e.name))},t.mark=function(t){return Object.setPrototypeOf?Object.setPrototypeOf(t,v):(t.__proto__=v,f(t,s,"GeneratorFunction")),t.prototype=Object.create(w),t},t.awrap=function(t){return{__await:t}},x(O.prototype),f(O.prototype,u,(function(){return this})),t.AsyncIterator=O,t.async=function(e,r,n,o,i){void 0===i&&(i=Promise);var a=new O(l(e,r,n,o),i);return t.isGeneratorFunction(r)?a:a.next().then((function(t){return t.done?t.value:a.next()}))},x(w),f(w,s,"Generator"),f(w,c,(function(){return this})),f(w,"toString",(function(){return"[object Generator]"})),t.keys=function(t){var e=Object(t),r=[];for(var n in e)r.push(n);return r.reverse(),function t(){for(;r.length;){var n=r.pop();if(n in e)return t.value=n,t.done=!1,t}return t.done=!0,t}},t.values=k,P.prototype={constructor:P,reset:function(t){if(this.prev=0,this.next=0,this.sent=this._sent=void 0,this.done=!1,this.delegate=null,this.method="next",this.arg=void 0,this.tryEntries.forEach(E),!t)for(var e in this)"t"===e.charAt(0)&&r.call(this,e)&&!isNaN(+e.slice(1))&&(this[e]=void 0)},stop:function(){this.done=!0;var t=this.tryEntries[0].completion;if("throw"===t.type)throw t.arg;return this.rval},dispatchException:function(t){if(this.done)throw t;var e=this;function n(r,n){return a.type="throw",a.arg=t,e.next=r,n&&(e.method="next",e.arg=void 0),!!n}for(var o=this.tryEntries.length-1;o>=0;--o){var i=this.tryEntries[o],a=i.completion;if("root"===i.tryLoc)return n("end");if(i.tryLoc<=this.prev){var c=r.call(i,"catchLoc"),u=r.call(i,"finallyLoc");if(c&&u){if(this.prev<i.catchLoc)return n(i.catchLoc,!0);if(this.prev<i.finallyLoc)return n(i.finallyLoc)}else if(c){if(this.prev<i.catchLoc)return n(i.catchLoc,!0)}else{if(!u)throw new Error("try statement without catch or finally");if(this.prev<i.finallyLoc)return n(i.finallyLoc)}}}},abrupt:function(t,e){for(var n=this.tryEntries.length-1;n>=0;--n){var o=this.tryEntries[n];if(o.tryLoc<=this.prev&&r.call(o,"finallyLoc")&&this.prev<o.finallyLoc){var i=o;break}}i&&("break"===t||"continue"===t)&&i.tryLoc<=e&&e<=i.finallyLoc&&(i=null);var a=i?i.completion:{};return a.type=t,a.arg=e,i?(this.method="next",this.next=i.finallyLoc,p):this.complete(a)},complete:function(t,e){if("throw"===t.type)throw t.arg;return"break"===t.type||"continue"===t.type?this.next=t.arg:"return"===t.type?(this.rval=this.arg=t.arg,this.method="return",this.next="end"):"normal"===t.type&&e&&(this.next=e),p},finish:function(t){for(var e=this.tryEntries.length-1;e>=0;--e){var r=this.tryEntries[e];if(r.finallyLoc===t)return this.complete(r.completion,r.afterLoc),E(r),p}},catch:function(t){for(var e=this.tryEntries.length-1;e>=0;--e){var r=this.tryEntries[e];if(r.tryLoc===t){var n=r.completion;if("throw"===n.type){var o=n.arg;E(r)}return o}}throw new Error("illegal catch attempt")},delegateYield:function(t,e,r){return this.delegate={iterator:k(t),resultName:e,nextLoc:r},"next"===this.method&&(this.arg=void 0),p}},t}function i(t){return function(t){if(Array.isArray(t))return a(t)}(t)||function(t){if("undefined"!=typeof Symbol&&null!=t[Symbol.iterator]||null!=t["@@iterator"])return Array.from(t)}(t)||function(t,e){if(t){if("string"==typeof t)return a(t,e);var r=Object.prototype.toString.call(t).slice(8,-1);return"Object"===r&&t.constructor&&(r=t.constructor.name),"Map"===r||"Set"===r?Array.from(t):"Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)?a(t,e):void 0}}(t)||function(){throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}function a(t,e){(null==e||e>t.length)&&(e=t.length);for(var r=0,n=new Array(e);r<e;r++)n[r]=t[r];return n}function c(t,e,r,n,o,i,a){try{var c=t[i](a),u=c.value}catch(t){return void r(t)}c.done?e(u):Promise.resolve(u).then(n,o)}function u(t){return function(){var e=this,r=arguments;return new Promise((function(n,o){var i=t.apply(e,r);function a(t){c(i,n,o,a,u,"next",t)}function u(t){c(i,n,o,a,u,"throw",t)}a(void 0)}))}}var s=r(550),f=r(258).loadPyodide,l=r(736);function h(t){postMessage([0,t])}function p(){return(p=u(o().mark((function t(e){var r,n,a,c,u,p,y,d,v,g,m,b,w,x,O,S,L,j,E,P,k,_;return o().wrap((function(t){for(;;)switch(t.prev=t.next){case 0:for(r=e.data_buffer,n=e.channels,a=e.model,(c=new s.EdfDecoder).setInput(r),c.decode(),u=c.getOutput(),p=[],y=0;y<u.getNumberOfSignals();y++)p.push(u.getSignalLabel(y));for(d=u.getNumberOfRecords(),v=[],g=[],m=0;m<n.length;m++)b=u.getPhysicalSignalConcatRecords(p.indexOf(n[m]),0,d),g.push(new Float32Array(b)),v.push(u.getSignalSamplingFrequency(p.indexOf(n[m])));return h(2),t.next=17,f({indexURL:"./pyodide"});case 17:return w=t.sent,t.prev=18,t.next=21,w.loadPackage("numpy");case 21:return t.next=23,w.loadPackage("scikit-learn");case 23:return t.next=25,w.loadPackage("scipy");case 25:return w.globals.set("channel_data",g),w.globals.set("sampling_rates",v),h(5),t.t0=w,t.next=31,fetch("./preprocessing.py");case 31:return t.next=33,t.sent.text();case 33:t.t1=t.sent,t.t0.runPython.call(t.t0,t.t1),t.next=41;break;case 37:return t.prev=37,t.t2=t.catch(18),console.log(t.t2),t.abrupt("return",[0]);case 41:return h(10),l.env.wasm.wasmPaths="./",t.next=45,l.InferenceSession.create("./models/".concat(a,".onnx"));case 45:x=t.sent,O=w.globals.get("samples").toJs(),S=[],L=0;case 49:if(!(L<O.length)){t.next=63;break}return j=new l.Tensor("float32",new Float32Array(O[L]),[1,11,n.length,3e3]),E={input:j},t.next=54,x.run(E);case 54:for(P=t.sent,k=P.output.data,_=0;_<k.length;_+=5)S.push(k.slice(_,_+5).indexOf(Math.max.apply(Math,i(k.slice(_,_+5)))));return h(10+Math.round(L/O.length*90)),t.next=60,new Promise((function(t){return setTimeout(t,0)}));case 60:L++,t.next=49;break;case 63:return h(0),t.abrupt("return",S);case 65:case"end":return t.stop()}}),t,null,[[18,37]])})))).apply(this,arguments)}onmessage=function(t){(function(t){return p.apply(this,arguments)})(t.data).then((function(t){postMessage([1,t])}))}}},n={};function o(t){var e=n[t];if(void 0!==e)return e.exports;var i=n[t]={exports:{}};return r[t].call(i.exports,i,i.exports,o),i.exports}return o.m=r,o.x=function(){var t=o.O(void 0,[84],(function(){return o(493)}));return o.O(t)},t=[],o.O=function(e,r,n,i){if(!r){var a=1/0;for(f=0;f<t.length;f++){r=t[f][0],n=t[f][1],i=t[f][2];for(var c=!0,u=0;u<r.length;u++)(!1&i||a>=i)&&Object.keys(o.O).every((function(t){return o.O[t](r[u])}))?r.splice(u--,1):(c=!1,i<a&&(a=i));if(c){t.splice(f--,1);var s=n();void 0!==s&&(e=s)}}return e}i=i||0;for(var f=t.length;f>0&&t[f-1][2]>i;f--)t[f]=t[f-1];t[f]=[r,n,i]},o.d=function(t,e){for(var r in e)o.o(e,r)&&!o.o(t,r)&&Object.defineProperty(t,r,{enumerable:!0,get:e[r]})},o.f={},o.e=function(t){return Promise.all(Object.keys(o.f).reduce((function(e,r){return o.f[r](t,e),e}),[]))},o.u=function(t){return t+".js"},o.g=function(){if("object"==typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(t){if("object"==typeof window)return window}}(),o.o=function(t,e){return Object.prototype.hasOwnProperty.call(t,e)},o.r=function(t){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(t,"__esModule",{value:!0})},function(){var t;o.g.importScripts&&(t=o.g.location+"");var e=o.g.document;if(!t&&e&&(e.currentScript&&(t=e.currentScript.src),!t)){var r=e.getElementsByTagName("script");r.length&&(t=r[r.length-1].src)}if(!t)throw new Error("Automatic publicPath is not supported in this browser");t=t.replace(/#.*$/,"").replace(/\?.*$/,"").replace(/\/[^\/]+$/,"/"),o.p=t}(),function(){var t={493:1};o.f.i=function(e,r){t[e]||importScripts(o.p+o.u(e))};var e=self.webpackChunkv4=self.webpackChunkv4||[],r=e.push.bind(e);e.push=function(e){var n=e[0],i=e[1],a=e[2];for(var c in i)o.o(i,c)&&(o.m[c]=i[c]);for(a&&a(o);n.length;)t[n.pop()]=1;r(e)}}(),e=o.x,o.x=function(){return o.e(84).then(e)},o.x()}()}));
//# sourceMappingURL=data:application/json;charset=utf-8;base64,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QUcvRCxFQUVBLElBQUk2QixFQUFxQnZJLEtBQXFCLGVBQUlBLEtBQXFCLGdCQUFLLEdBQ3hFd0ksRUFBNkJELEVBQW1CaEgsS0FBS2tILEtBQUtGLEdBQzlEQSxFQUFtQmhILEtBekJBLFNBQVN3QyxHQUMzQixJQUFJd0IsRUFBV3hCLEVBQUssR0FDaEIyRSxFQUFjM0UsRUFBSyxHQUNuQjRFLEVBQVU1RSxFQUFLLEdBQ25CLElBQUksSUFBSWUsS0FBWTRELEVBQ2hCN0QsRUFBb0J3QixFQUFFcUMsRUFBYTVELEtBQ3JDRCxFQUFvQk0sRUFBRUwsR0FBWTRELEVBQVk1RCxJQUloRCxJQURHNkQsR0FBU0EsRUFBUTlELEdBQ2RVLEVBQVN6RCxRQUNkd0csRUFBZ0IvQyxFQUFTcUQsT0FBUyxFQUNuQ0osRUFBMkJ6RSxFQUM1QixLVnRCSTdELEVBQU8yRSxFQUFvQk8sRUFDL0JQLEVBQW9CTyxFQUFJLFdBQ3ZCLE9BQU9QLEVBQW9CNEIsRUFBRSxJQUFJL0IsS0FBS3hFLEVBQ3ZDLEVXRjBCMkUsRUFBb0JPIiwic291cmNlcyI6WyJ3ZWJwYWNrOi8vdjQvd2VicGFjay91bml2ZXJzYWxNb2R1bGVEZWZpbml0aW9uIiwid2VicGFjazovL3Y0L3dlYnBhY2svcnVudGltZS9jaHVuayBsb2FkZWQiLCJ3ZWJwYWNrOi8vdjQvd2VicGFjay9ydW50aW1lL3N0YXJ0dXAgY2h1bmsgZGVwZW5kZW5jaWVzIiwid2VicGFjazovL3Y0Ly4vc3JjL29ydF93ZWJ3b3JrZXIuanMiLCJ3ZWJwYWNrOi8vdjQvd2VicGFjay9ib290c3RyYXAiLCJ3ZWJwYWNrOi8vdjQvd2VicGFjay9ydW50aW1lL2RlZmluZSBwcm9wZXJ0eSBnZXR0ZXJzIiwid2VicGFjazovL3Y0L3dlYnBhY2svcnVudGltZS9lbnN1cmUgY2h1bmsiLCJ3ZWJwYWNrOi8vdjQvd2VicGFjay9ydW50aW1lL2dldCBqYXZhc2NyaXB0IGNodW5rIGZpbGVuYW1lIiwid2VicGFjazovL3Y0L3dlYnBhY2svcnVudGltZS9nbG9iYWwiLCJ3ZWJwYWNrOi8vdjQvd2VicGFjay9ydW50aW1lL2hhc093blByb3BlcnR5IHNob3J0aGFuZCIsIndlYnBhY2s6Ly92NC93ZWJwYWNrL3J1bnRpbWUvbWFrZSBuYW1lc3BhY2Ugb2JqZWN0Iiwid2VicGFjazovL3Y0L3dlYnBhY2svcnVudGltZS9wdWJsaWNQYXRoIiwid2VicGFjazovL3Y0L3dlYnBhY2svcnVudGltZS9pbXBvcnRTY3JpcHRzIGNodW5rIGxvYWRpbmciLCJ3ZWJwYWNrOi8vdjQvd2VicGFjay9zdGFydHVwIl0sInNvdXJjZXNDb250ZW50IjpbIihmdW5jdGlvbiB3ZWJwYWNrVW5pdmVyc2FsTW9kdWxlRGVmaW5pdGlvbihyb290LCBmYWN0b3J5KSB7XG5cdGlmKHR5cGVvZiBleHBvcnRzID09PSAnb2JqZWN0JyAmJiB0eXBlb2YgbW9kdWxlID09PSAnb2JqZWN0Jylcblx0XHRtb2R1bGUuZXhwb3J0cyA9IGZhY3RvcnkoKTtcblx0ZWxzZSBpZih0eXBlb2YgZGVmaW5lID09PSAnZnVuY3Rpb24nICYmIGRlZmluZS5hbWQpXG5cdFx0ZGVmaW5lKFtdLCBmYWN0b3J5KTtcblx0ZWxzZSB7XG5cdFx0dmFyIGEgPSBmYWN0b3J5KCk7XG5cdFx0Zm9yKHZhciBpIGluIGEpICh0eXBlb2YgZXhwb3J0cyA9PT0gJ29iamVjdCcgPyBleHBvcnRzIDogcm9vdClbaV0gPSBhW2ldO1xuXHR9XG59KShzZWxmLCBmdW5jdGlvbigpIHtcbnJldHVybiAiLCJ2YXIgZGVmZXJyZWQgPSBbXTtcbl9fd2VicGFja19yZXF1aXJlX18uTyA9IGZ1bmN0aW9uKHJlc3VsdCwgY2h1bmtJZHMsIGZuLCBwcmlvcml0eSkge1xuXHRpZihjaHVua0lkcykge1xuXHRcdHByaW9yaXR5ID0gcHJpb3JpdHkgfHwgMDtcblx0XHRmb3IodmFyIGkgPSBkZWZlcnJlZC5sZW5ndGg7IGkgPiAwICYmIGRlZmVycmVkW2kgLSAxXVsyXSA+IHByaW9yaXR5OyBpLS0pIGRlZmVycmVkW2ldID0gZGVmZXJyZWRbaSAtIDFdO1xuXHRcdGRlZmVycmVkW2ldID0gW2NodW5rSWRzLCBmbiwgcHJpb3JpdHldO1xuXHRcdHJldHVybjtcblx0fVxuXHR2YXIgbm90RnVsZmlsbGVkID0gSW5maW5pdHk7XG5cdGZvciAodmFyIGkgPSAwOyBpIDwgZGVmZXJyZWQubGVuZ3RoOyBpKyspIHtcblx0XHR2YXIgY2h1bmtJZHMgPSBkZWZlcnJlZFtpXVswXTtcblx0XHR2YXIgZm4gPSBkZWZlcnJlZFtpXVsxXTtcblx0XHR2YXIgcHJpb3JpdHkgPSBkZWZlcnJlZFtpXVsyXTtcblx0XHR2YXIgZnVsZmlsbGVkID0gdHJ1ZTtcblx0XHRmb3IgKHZhciBqID0gMDsgaiA8IGNodW5rSWRzLmxlbmd0aDsgaisrKSB7XG5cdFx0XHRpZiAoKHByaW9yaXR5ICYgMSA9PT0gMCB8fCBub3RGdWxmaWxsZWQgPj0gcHJpb3JpdHkpICYmIE9iamVjdC5rZXlzKF9fd2VicGFja19yZXF1aXJlX18uTykuZXZlcnkoZnVuY3Rpb24oa2V5KSB7IHJldHVybiBfX3dlYnBhY2tfcmVxdWlyZV9fLk9ba2V5XShjaHVua0lkc1tqXSk7IH0pKSB7XG5cdFx0XHRcdGNodW5rSWRzLnNwbGljZShqLS0sIDEpO1xuXHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0ZnVsZmlsbGVkID0gZmFsc2U7XG5cdFx0XHRcdGlmKHByaW9yaXR5IDwgbm90RnVsZmlsbGVkKSBub3RGdWxmaWxsZWQgPSBwcmlvcml0eTtcblx0XHRcdH1cblx0XHR9XG5cdFx0aWYoZnVsZmlsbGVkKSB7XG5cdFx0XHRkZWZlcnJlZC5zcGxpY2UoaS0tLCAxKVxuXHRcdFx0dmFyIHIgPSBmbigpO1xuXHRcdFx0aWYgKHIgIT09IHVuZGVmaW5lZCkgcmVzdWx0ID0gcjtcblx0XHR9XG5cdH1cblx0cmV0dXJuIHJlc3VsdDtcbn07IiwidmFyIG5leHQgPSBfX3dlYnBhY2tfcmVxdWlyZV9fLng7XG5fX3dlYnBhY2tfcmVxdWlyZV9fLnggPSBmdW5jdGlvbigpIHtcblx0cmV0dXJuIF9fd2VicGFja19yZXF1aXJlX18uZSg4NCkudGhlbihuZXh0KTtcbn07IiwiY29uc3QgZWRmZGVjb2RlciA9IHJlcXVpcmUoJ2VkZmRlY29kZXInKTtcbmNvbnN0IHsgbG9hZFB5b2RpZGUgfSA9IHJlcXVpcmUoJ3B5b2RpZGUnKTtcbmNvbnN0IG9ydCA9IHJlcXVpcmUoJ29ubnhydW50aW1lLXdlYicpO1xuXG5mdW5jdGlvbiB1cGRhdGVQcm9ncmVzcyhwcm9ncmVzcykge1xuICBwb3N0TWVzc2FnZShbMCwgcHJvZ3Jlc3NdKTtcbn1cblxuYXN5bmMgZnVuY3Rpb24gb3J0QmFja2dyb3VuZChhcmdzKSB7XG4gIC8vIGNvbnNvbGUubG9nKGFyZ3MpO1xuICBjb25zdCBkYXRhQnVmZmVyID0gYXJncy5kYXRhX2J1ZmZlcjtcbiAgY29uc3QgeyBjaGFubmVscyB9ID0gYXJncztcbiAgY29uc3QgeyBtb2RlbCB9ID0gYXJncztcbiAgY29uc3QgYmF0Y2hTaXplID0gMTA7XG5cbiAgLy8gdHJ5IHtcbiAgLy8gbG9hZCBkYXRhXG4gIGNvbnN0IGRlY29kZXIgPSBuZXcgZWRmZGVjb2Rlci5FZGZEZWNvZGVyKCk7XG4gIGRlY29kZXIuc2V0SW5wdXQoZGF0YUJ1ZmZlcik7XG4gIGRlY29kZXIuZGVjb2RlKCk7XG4gIGNvbnN0IG91dHB1dCA9IGRlY29kZXIuZ2V0T3V0cHV0KCk7XG4gIGNvbnN0IGNoYW5uZWxOYW1lcyA9IFtdO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IG91dHB1dC5nZXROdW1iZXJPZlNpZ25hbHMoKTsgaSsrKSB7XG4gICAgY2hhbm5lbE5hbWVzLnB1c2gob3V0cHV0LmdldFNpZ25hbExhYmVsKGkpKTtcbiAgfVxuXG4gIC8vIGdldCB0aGUgbnVtYmVyIG9mIHJlY29yZHNcbiAgLy8gbm90ZSB0aGF0IGVhY2ggc2lnbmFsIGNhbiBoYXZlIG11bHRpcGxlIHJlY29yZHMuIEEgY2xhc3NpYyBjYXNlIGlzIHRvIGhhdmUgMS1zZWNvbmQtbG9uZyByZWNvcmRzXG4gIGNvbnN0IG51bWJlck9mUmVjb3JkcyA9IG91dHB1dC5nZXROdW1iZXJPZlJlY29yZHMoKTtcbiAgY29uc3Qgc2FtcGxpbmdSYXRlcyA9IFtdO1xuICBjb25zdCBjaGFubmVsRGF0YSA9IFtdO1xuXG4gIGZvciAobGV0IGNoID0gMDsgY2ggPCBjaGFubmVscy5sZW5ndGg7IGNoKyspIHtcbiAgICAvLyBJdCBjYW4gYmUgY29udmVuaWVudCB0byBjb25jYXRlbmF0ZSByZWNvcmRzIGZyb20gYSBzYW1lIHNpZ25hbFxuICAgIC8vIGZvciBleGFtcGxlIHRvIGdldCBhIHNpZ25hbCB0aGF0IGlzIGxvbmdlciB0aGFuIDFzZWNcbiAgICBjb25zdCBzaWduYWwgPSBvdXRwdXQuZ2V0UGh5c2ljYWxTaWduYWxDb25jYXRSZWNvcmRzKFxuICAgICAgY2hhbm5lbE5hbWVzLmluZGV4T2YoY2hhbm5lbHNbY2hdKSxcbiAgICAgIDAsXG4gICAgICBudW1iZXJPZlJlY29yZHNcbiAgICApO1xuICAgIC8vIGVzbGludC1kaXNhYmxlLW5leHQtbGluZSBuby11bmRlZlxuICAgIGNoYW5uZWxEYXRhLnB1c2gobmV3IEZsb2F0MzJBcnJheShzaWduYWwpKTtcbiAgICBzYW1wbGluZ1JhdGVzLnB1c2goXG4gICAgICBvdXRwdXQuZ2V0U2lnbmFsU2FtcGxpbmdGcmVxdWVuY3koY2hhbm5lbE5hbWVzLmluZGV4T2YoY2hhbm5lbHNbY2hdKSlcbiAgICApO1xuICB9XG4gIHVwZGF0ZVByb2dyZXNzKDIpO1xuXG4gIC8vIGNvbnN0IHB5b2RpZGUgPSBhd2FpdCBsb2FkUHlvZGlkZSgpO1xuICBjb25zdCBweW9kaWRlID0gYXdhaXQgbG9hZFB5b2RpZGUoeyBpbmRleFVSTDogJy4vcHlvZGlkZScgfSk7XG4gIHRyeSB7XG4gICAgYXdhaXQgcHlvZGlkZS5sb2FkUGFja2FnZSgnbnVtcHknKTtcbiAgICBhd2FpdCBweW9kaWRlLmxvYWRQYWNrYWdlKCdzY2lraXQtbGVhcm4nKTtcbiAgICBhd2FpdCBweW9kaWRlLmxvYWRQYWNrYWdlKCdzY2lweScpO1xuICAgIC8vIHB5b2RpZGUuRlMubWtkaXIoXCIvbG9jYWxfZGlyZWN0b3J5XCIpO1xuICAgIC8vIHB5b2RpZGUuRlMubW91bnQocHlvZGlkZS5GUy5maWxlc3lzdGVtcy5OT0RFRlMsIHtyb290OiBcIi9ob21lL25pa2xhcy9yZXBvcy9jb2RlL3NsZWVwX2xhYmVsLW5vaXNlL2NhY2hlL3NsZWVwZWRmeC9cIn0sIFwiL2xvY2FsX2RpcmVjdG9yeVwiKTtcbiAgICBweW9kaWRlLmdsb2JhbHMuc2V0KCdjaGFubmVsX2RhdGEnLCBjaGFubmVsRGF0YSk7XG4gICAgcHlvZGlkZS5nbG9iYWxzLnNldCgnc2FtcGxpbmdfcmF0ZXMnLCBzYW1wbGluZ1JhdGVzKTtcbiAgICB1cGRhdGVQcm9ncmVzcyg1KTtcbiAgICBweW9kaWRlLnJ1blB5dGhvbihhd2FpdCAoYXdhaXQgZmV0Y2goJy4vcHJlcHJvY2Vzc2luZy5weScpKS50ZXh0KCkpO1xuICB9IGNhdGNoIChlcnIpIHtcbiAgICBjb25zb2xlLmxvZyhlcnIpO1xuICAgIHJldHVybiBbMF07XG4gIH1cbiAgdXBkYXRlUHJvZ3Jlc3MoMTApO1xuXG4gIC8vc2V0dXAgb3J0XG4gIC8vIG92ZXJyaWRlIHBhdGggb2Ygd2FzbSBmaWxlcyAtIHVzaW5nIGEgcHJlZml4XG4gIC8vIGluIHRoaXMgZXhhbXBsZSwgT05OWCBSdW50aW1lIFdlYiB3aWxsIHRyeSB0byBsb2FkIGZpbGUgZnJvbSBodHRwczovL2V4YW1wbGUuY29tL215LWV4YW1wbGUvb3J0LXdhc20qLndhc21cbiAgLy8gb3J0LmVudi53YXNtLndhc21QYXRocyA9ICdodHRwczovL2V4YW1wbGUuY29tL215LWV4YW1wbGUvJztcbiAgb3J0LmVudi53YXNtLndhc21QYXRocyA9ICcuLyc7XG4gIC8vIGNyZWF0ZSBhIG5ldyBzZXNzaW9uIGFuZCBsb2FkIHRoZSBzcGVjaWZpYyBtb2RlbC5cbiAgY29uc3Qgc2Vzc2lvbiA9IGF3YWl0IG9ydC5JbmZlcmVuY2VTZXNzaW9uLmNyZWF0ZShgLi9tb2RlbHMvJHttb2RlbH0ub25ueGApO1xuXG4gIC8vIHByZXBhcmUgaW5wdXRzLiBhIHRlbnNvciBuZWVkIGl0cyBjb3JyZXNwb25kaW5nIFR5cGVkQXJyYXkgYXMgZGF0YVxuICBjb25zdCBweURhdGEgPSBweW9kaWRlLmdsb2JhbHMuZ2V0KCdzYW1wbGVzJykudG9KcygpO1xuICBjb25zdCBsYWJlbHMgPSBbXTtcblxuICBmb3IgKGxldCBpID0gMDsgaSA8IHB5RGF0YS5sZW5ndGg7IGkrKykge1xuICAgIC8vIGVzbGludC1kaXNhYmxlLW5leHQtbGluZSBuby11bmRlZlxuICAgIGNvbnN0IHRlbnNvciA9IG5ldyBvcnQuVGVuc29yKCdmbG9hdDMyJywgbmV3IEZsb2F0MzJBcnJheShweURhdGFbaV0pLCBbXG4gICAgICAxLFxuICAgICAgMTEsXG4gICAgICBjaGFubmVscy5sZW5ndGgsXG4gICAgICAzMDAwLFxuICAgIF0pO1xuICAgIC8vIHByZXBhcmUgZmVlZHMuIHVzZSBtb2RlbCBpbnB1dCBuYW1lcyBhcyBrZXlzLlxuICAgIGNvbnN0IGZlZWRzID0geyBpbnB1dDogdGVuc29yIH07XG4gICAgLy8gZmVlZCBpbnB1dHMgYW5kIHJ1blxuICAgIGNvbnN0IHJlc3VsdHMgPSBhd2FpdCBzZXNzaW9uLnJ1bihmZWVkcyk7XG4gICAgLy8gcmVhZCBmcm9tIHJlc3VsdHNcbiAgICBjb25zdCBkYXRhT3V0ID0gcmVzdWx0cy5vdXRwdXQuZGF0YTtcbiAgICAvLyBjb25zb2xlLmxvZyhgZGF0YSBvZiByZXN1bHQgdGVuc29yOiAke2RhdGFfb3V0fWApO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgZGF0YU91dC5sZW5ndGg7IGkgKz0gNSkge1xuICAgICAgbGFiZWxzLnB1c2goXG4gICAgICAgIGRhdGFPdXQuc2xpY2UoaSwgaSArIDUpLmluZGV4T2YoTWF0aC5tYXgoLi4uZGF0YU91dC5zbGljZShpLCBpICsgNSkpKVxuICAgICAgKTtcbiAgICB9XG5cbiAgICB1cGRhdGVQcm9ncmVzcygxMCArIE1hdGgucm91bmQoKGkgLyBweURhdGEubGVuZ3RoKSAqIDkwKSk7XG4gICAgLy8gZXNsaW50LWRpc2FibGUtbmV4dC1saW5lIG5vLXVuZGVmXG4gICAgYXdhaXQgbmV3IFByb21pc2UoKHIpID0+IHNldFRpbWVvdXQociwgMCkpO1xuICB9XG4gIHVwZGF0ZVByb2dyZXNzKDApO1xuXG4gIHJldHVybiBsYWJlbHM7XG5cbiAgLy8gfSBjYXRjaCAoZSkge1xuICAvLyAgICAgY29uc29sZS5lcnJvcihgZmFpbGVkIHRvIGluZmVyZW5jZSBPTk5YIG1vZGVsOiAke2V9LmApO1xuICAvLyB9XG59XG5cbm9ubWVzc2FnZSA9IGZ1bmN0aW9uIChldmVudCkge1xuICAvL1RoaXMgd2lsbCBiZSBjYWxsZWQgd2hlbiB3b3JrZXIucG9zdE1lc3NhZ2UgaXMgY2FsbGVkIGluIHRoZSBvdXRzaWRlIGNvZGUuXG4gIG9ydEJhY2tncm91bmQoZXZlbnQuZGF0YSkudGhlbigocmVzdWx0KSA9PiB7XG4gICAgcG9zdE1lc3NhZ2UoWzEsIHJlc3VsdF0pOyAvL1NlbmQgdGhlIHJlc3VsdCB0byB0aGUgb3V0c2lkZSBjb2RlLlxuICB9KTtcbn07XG5cbi8vIGV4cG9ydCB7IG9ydEJhY2tncm91bmQgfTtcbiIsIi8vIFRoZSBtb2R1bGUgY2FjaGVcbnZhciBfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX18gPSB7fTtcblxuLy8gVGhlIHJlcXVpcmUgZnVuY3Rpb25cbmZ1bmN0aW9uIF9fd2VicGFja19yZXF1aXJlX18obW9kdWxlSWQpIHtcblx0Ly8gQ2hlY2sgaWYgbW9kdWxlIGlzIGluIGNhY2hlXG5cdHZhciBjYWNoZWRNb2R1bGUgPSBfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX19bbW9kdWxlSWRdO1xuXHRpZiAoY2FjaGVkTW9kdWxlICE9PSB1bmRlZmluZWQpIHtcblx0XHRyZXR1cm4gY2FjaGVkTW9kdWxlLmV4cG9ydHM7XG5cdH1cblx0Ly8gQ3JlYXRlIGEgbmV3IG1vZHVsZSAoYW5kIHB1dCBpdCBpbnRvIHRoZSBjYWNoZSlcblx0dmFyIG1vZHVsZSA9IF9fd2VicGFja19tb2R1bGVfY2FjaGVfX1ttb2R1bGVJZF0gPSB7XG5cdFx0Ly8gbm8gbW9kdWxlLmlkIG5lZWRlZFxuXHRcdC8vIG5vIG1vZHVsZS5sb2FkZWQgbmVlZGVkXG5cdFx0ZXhwb3J0czoge31cblx0fTtcblxuXHQvLyBFeGVjdXRlIHRoZSBtb2R1bGUgZnVuY3Rpb25cblx0X193ZWJwYWNrX21vZHVsZXNfX1ttb2R1bGVJZF0uY2FsbChtb2R1bGUuZXhwb3J0cywgbW9kdWxlLCBtb2R1bGUuZXhwb3J0cywgX193ZWJwYWNrX3JlcXVpcmVfXyk7XG5cblx0Ly8gUmV0dXJuIHRoZSBleHBvcnRzIG9mIHRoZSBtb2R1bGVcblx0cmV0dXJuIG1vZHVsZS5leHBvcnRzO1xufVxuXG4vLyBleHBvc2UgdGhlIG1vZHVsZXMgb2JqZWN0IChfX3dlYnBhY2tfbW9kdWxlc19fKVxuX193ZWJwYWNrX3JlcXVpcmVfXy5tID0gX193ZWJwYWNrX21vZHVsZXNfXztcblxuLy8gdGhlIHN0YXJ0dXAgZnVuY3Rpb25cbl9fd2VicGFja19yZXF1aXJlX18ueCA9IGZ1bmN0aW9uKCkge1xuXHQvLyBMb2FkIGVudHJ5IG1vZHVsZSBhbmQgcmV0dXJuIGV4cG9ydHNcblx0Ly8gVGhpcyBlbnRyeSBtb2R1bGUgZGVwZW5kcyBvbiBvdGhlciBsb2FkZWQgY2h1bmtzIGFuZCBleGVjdXRpb24gbmVlZCB0byBiZSBkZWxheWVkXG5cdHZhciBfX3dlYnBhY2tfZXhwb3J0c19fID0gX193ZWJwYWNrX3JlcXVpcmVfXy5PKHVuZGVmaW5lZCwgWzg0XSwgZnVuY3Rpb24oKSB7IHJldHVybiBfX3dlYnBhY2tfcmVxdWlyZV9fKDQ5Myk7IH0pXG5cdF9fd2VicGFja19leHBvcnRzX18gPSBfX3dlYnBhY2tfcmVxdWlyZV9fLk8oX193ZWJwYWNrX2V4cG9ydHNfXyk7XG5cdHJldHVybiBfX3dlYnBhY2tfZXhwb3J0c19fO1xufTtcblxuIiwiLy8gZGVmaW5lIGdldHRlciBmdW5jdGlvbnMgZm9yIGhhcm1vbnkgZXhwb3J0c1xuX193ZWJwYWNrX3JlcXVpcmVfXy5kID0gZnVuY3Rpb24oZXhwb3J0cywgZGVmaW5pdGlvbikge1xuXHRmb3IodmFyIGtleSBpbiBkZWZpbml0aW9uKSB7XG5cdFx0aWYoX193ZWJwYWNrX3JlcXVpcmVfXy5vKGRlZmluaXRpb24sIGtleSkgJiYgIV9fd2VicGFja19yZXF1aXJlX18ubyhleHBvcnRzLCBrZXkpKSB7XG5cdFx0XHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywga2V5LCB7IGVudW1lcmFibGU6IHRydWUsIGdldDogZGVmaW5pdGlvbltrZXldIH0pO1xuXHRcdH1cblx0fVxufTsiLCJfX3dlYnBhY2tfcmVxdWlyZV9fLmYgPSB7fTtcbi8vIFRoaXMgZmlsZSBjb250YWlucyBvbmx5IHRoZSBlbnRyeSBjaHVuay5cbi8vIFRoZSBjaHVuayBsb2FkaW5nIGZ1bmN0aW9uIGZvciBhZGRpdGlvbmFsIGNodW5rc1xuX193ZWJwYWNrX3JlcXVpcmVfXy5lID0gZnVuY3Rpb24oY2h1bmtJZCkge1xuXHRyZXR1cm4gUHJvbWlzZS5hbGwoT2JqZWN0LmtleXMoX193ZWJwYWNrX3JlcXVpcmVfXy5mKS5yZWR1Y2UoZnVuY3Rpb24ocHJvbWlzZXMsIGtleSkge1xuXHRcdF9fd2VicGFja19yZXF1aXJlX18uZltrZXldKGNodW5rSWQsIHByb21pc2VzKTtcblx0XHRyZXR1cm4gcHJvbWlzZXM7XG5cdH0sIFtdKSk7XG59OyIsIi8vIFRoaXMgZnVuY3Rpb24gYWxsb3cgdG8gcmVmZXJlbmNlIGFzeW5jIGNodW5rcyBhbmQgc2libGluZyBjaHVua3MgZm9yIHRoZSBlbnRyeXBvaW50XG5fX3dlYnBhY2tfcmVxdWlyZV9fLnUgPSBmdW5jdGlvbihjaHVua0lkKSB7XG5cdC8vIHJldHVybiB1cmwgZm9yIGZpbGVuYW1lcyBiYXNlZCBvbiB0ZW1wbGF0ZVxuXHRyZXR1cm4gXCJcIiArIGNodW5rSWQgKyBcIi5qc1wiO1xufTsiLCJfX3dlYnBhY2tfcmVxdWlyZV9fLmcgPSAoZnVuY3Rpb24oKSB7XG5cdGlmICh0eXBlb2YgZ2xvYmFsVGhpcyA9PT0gJ29iamVjdCcpIHJldHVybiBnbG9iYWxUaGlzO1xuXHR0cnkge1xuXHRcdHJldHVybiB0aGlzIHx8IG5ldyBGdW5jdGlvbigncmV0dXJuIHRoaXMnKSgpO1xuXHR9IGNhdGNoIChlKSB7XG5cdFx0aWYgKHR5cGVvZiB3aW5kb3cgPT09ICdvYmplY3QnKSByZXR1cm4gd2luZG93O1xuXHR9XG59KSgpOyIsIl9fd2VicGFja19yZXF1aXJlX18ubyA9IGZ1bmN0aW9uKG9iaiwgcHJvcCkgeyByZXR1cm4gT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKG9iaiwgcHJvcCk7IH0iLCIvLyBkZWZpbmUgX19lc01vZHVsZSBvbiBleHBvcnRzXG5fX3dlYnBhY2tfcmVxdWlyZV9fLnIgPSBmdW5jdGlvbihleHBvcnRzKSB7XG5cdGlmKHR5cGVvZiBTeW1ib2wgIT09ICd1bmRlZmluZWQnICYmIFN5bWJvbC50b1N0cmluZ1RhZykge1xuXHRcdE9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBTeW1ib2wudG9TdHJpbmdUYWcsIHsgdmFsdWU6ICdNb2R1bGUnIH0pO1xuXHR9XG5cdE9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCAnX19lc01vZHVsZScsIHsgdmFsdWU6IHRydWUgfSk7XG59OyIsInZhciBzY3JpcHRVcmw7XG5pZiAoX193ZWJwYWNrX3JlcXVpcmVfXy5nLmltcG9ydFNjcmlwdHMpIHNjcmlwdFVybCA9IF9fd2VicGFja19yZXF1aXJlX18uZy5sb2NhdGlvbiArIFwiXCI7XG52YXIgZG9jdW1lbnQgPSBfX3dlYnBhY2tfcmVxdWlyZV9fLmcuZG9jdW1lbnQ7XG5pZiAoIXNjcmlwdFVybCAmJiBkb2N1bWVudCkge1xuXHRpZiAoZG9jdW1lbnQuY3VycmVudFNjcmlwdClcblx0XHRzY3JpcHRVcmwgPSBkb2N1bWVudC5jdXJyZW50U2NyaXB0LnNyY1xuXHRpZiAoIXNjcmlwdFVybCkge1xuXHRcdHZhciBzY3JpcHRzID0gZG9jdW1lbnQuZ2V0RWxlbWVudHNCeVRhZ05hbWUoXCJzY3JpcHRcIik7XG5cdFx0aWYoc2NyaXB0cy5sZW5ndGgpIHNjcmlwdFVybCA9IHNjcmlwdHNbc2NyaXB0cy5sZW5ndGggLSAxXS5zcmNcblx0fVxufVxuLy8gV2hlbiBzdXBwb3J0aW5nIGJyb3dzZXJzIHdoZXJlIGFuIGF1dG9tYXRpYyBwdWJsaWNQYXRoIGlzIG5vdCBzdXBwb3J0ZWQgeW91IG11c3Qgc3BlY2lmeSBhbiBvdXRwdXQucHVibGljUGF0aCBtYW51YWxseSB2aWEgY29uZmlndXJhdGlvblxuLy8gb3IgcGFzcyBhbiBlbXB0eSBzdHJpbmcgKFwiXCIpIGFuZCBzZXQgdGhlIF9fd2VicGFja19wdWJsaWNfcGF0aF9fIHZhcmlhYmxlIGZyb20geW91ciBjb2RlIHRvIHVzZSB5b3VyIG93biBsb2dpYy5cbmlmICghc2NyaXB0VXJsKSB0aHJvdyBuZXcgRXJyb3IoXCJBdXRvbWF0aWMgcHVibGljUGF0aCBpcyBub3Qgc3VwcG9ydGVkIGluIHRoaXMgYnJvd3NlclwiKTtcbnNjcmlwdFVybCA9IHNjcmlwdFVybC5yZXBsYWNlKC8jLiokLywgXCJcIikucmVwbGFjZSgvXFw/LiokLywgXCJcIikucmVwbGFjZSgvXFwvW15cXC9dKyQvLCBcIi9cIik7XG5fX3dlYnBhY2tfcmVxdWlyZV9fLnAgPSBzY3JpcHRVcmw7IiwiLy8gbm8gYmFzZVVSSVxuXG4vLyBvYmplY3QgdG8gc3RvcmUgbG9hZGVkIGNodW5rc1xuLy8gXCIxXCIgbWVhbnMgXCJhbHJlYWR5IGxvYWRlZFwiXG52YXIgaW5zdGFsbGVkQ2h1bmtzID0ge1xuXHQ0OTM6IDFcbn07XG5cbi8vIGltcG9ydFNjcmlwdHMgY2h1bmsgbG9hZGluZ1xudmFyIGluc3RhbGxDaHVuayA9IGZ1bmN0aW9uKGRhdGEpIHtcblx0dmFyIGNodW5rSWRzID0gZGF0YVswXTtcblx0dmFyIG1vcmVNb2R1bGVzID0gZGF0YVsxXTtcblx0dmFyIHJ1bnRpbWUgPSBkYXRhWzJdO1xuXHRmb3IodmFyIG1vZHVsZUlkIGluIG1vcmVNb2R1bGVzKSB7XG5cdFx0aWYoX193ZWJwYWNrX3JlcXVpcmVfXy5vKG1vcmVNb2R1bGVzLCBtb2R1bGVJZCkpIHtcblx0XHRcdF9fd2VicGFja19yZXF1aXJlX18ubVttb2R1bGVJZF0gPSBtb3JlTW9kdWxlc1ttb2R1bGVJZF07XG5cdFx0fVxuXHR9XG5cdGlmKHJ1bnRpbWUpIHJ1bnRpbWUoX193ZWJwYWNrX3JlcXVpcmVfXyk7XG5cdHdoaWxlKGNodW5rSWRzLmxlbmd0aClcblx0XHRpbnN0YWxsZWRDaHVua3NbY2h1bmtJZHMucG9wKCldID0gMTtcblx0cGFyZW50Q2h1bmtMb2FkaW5nRnVuY3Rpb24oZGF0YSk7XG59O1xuX193ZWJwYWNrX3JlcXVpcmVfXy5mLmkgPSBmdW5jdGlvbihjaHVua0lkLCBwcm9taXNlcykge1xuXHQvLyBcIjFcIiBpcyB0aGUgc2lnbmFsIGZvciBcImFscmVhZHkgbG9hZGVkXCJcblx0aWYoIWluc3RhbGxlZENodW5rc1tjaHVua0lkXSkge1xuXHRcdGlmKHRydWUpIHsgLy8gYWxsIGNodW5rcyBoYXZlIEpTXG5cdFx0XHRpbXBvcnRTY3JpcHRzKF9fd2VicGFja19yZXF1aXJlX18ucCArIF9fd2VicGFja19yZXF1aXJlX18udShjaHVua0lkKSk7XG5cdFx0fVxuXHR9XG59O1xuXG52YXIgY2h1bmtMb2FkaW5nR2xvYmFsID0gc2VsZltcIndlYnBhY2tDaHVua3Y0XCJdID0gc2VsZltcIndlYnBhY2tDaHVua3Y0XCJdIHx8IFtdO1xudmFyIHBhcmVudENodW5rTG9hZGluZ0Z1bmN0aW9uID0gY2h1bmtMb2FkaW5nR2xvYmFsLnB1c2guYmluZChjaHVua0xvYWRpbmdHbG9iYWwpO1xuY2h1bmtMb2FkaW5nR2xvYmFsLnB1c2ggPSBpbnN0YWxsQ2h1bms7XG5cbi8vIG5vIEhNUlxuXG4vLyBubyBITVIgbWFuaWZlc3QiLCIvLyBydW4gc3RhcnR1cFxudmFyIF9fd2VicGFja19leHBvcnRzX18gPSBfX3dlYnBhY2tfcmVxdWlyZV9fLngoKTtcbiJdLCJuYW1lcyI6WyJyb290IiwiZmFjdG9yeSIsImV4cG9ydHMiLCJtb2R1bGUiLCJkZWZpbmUiLCJhbWQiLCJhIiwiaSIsInNlbGYiLCJkZWZlcnJlZCIsIm5leHQiLCJlZGZkZWNvZGVyIiwicmVxdWlyZSIsImxvYWRQeW9kaWRlIiwib3J0IiwidXBkYXRlUHJvZ3Jlc3MiLCJwcm9ncmVzcyIsInBvc3RNZXNzYWdlIiwiYXJncyIsImRhdGFCdWZmZXIiLCJkYXRhX2J1ZmZlciIsImNoYW5uZWxzIiwibW9kZWwiLCJkZWNvZGVyIiwiRWRmRGVjb2RlciIsInNldElucHV0IiwiZGVjb2RlIiwib3V0cHV0IiwiZ2V0T3V0cHV0IiwiY2hhbm5lbE5hbWVzIiwiZ2V0TnVtYmVyT2ZTaWduYWxzIiwicHVzaCIsImdldFNpZ25hbExhYmVsIiwibnVtYmVyT2ZSZWNvcmRzIiwiZ2V0TnVtYmVyT2ZSZWNvcmRzIiwic2FtcGxpbmdSYXRlcyIsImNoYW5uZWxEYXRhIiwiY2giLCJsZW5ndGgiLCJzaWduYWwiLCJnZXRQaHlzaWNhbFNpZ25hbENvbmNhdFJlY29yZHMiLCJpbmRleE9mIiwiRmxvYXQzMkFycmF5IiwiZ2V0U2lnbmFsU2FtcGxpbmdGcmVxdWVuY3kiLCJpbmRleFVSTCIsInB5b2RpZGUiLCJsb2FkUGFja2FnZSIsImdsb2JhbHMiLCJzZXQiLCJmZXRjaCIsInRleHQiLCJydW5QeXRob24iLCJjb25zb2xlIiwibG9nIiwiZW52Iiwid2FzbSIsIndhc21QYXRocyIsIkluZmVyZW5jZVNlc3Npb24iLCJjcmVhdGUiLCJzZXNzaW9uIiwicHlEYXRhIiwiZ2V0IiwidG9KcyIsImxhYmVscyIsInRlbnNvciIsIlRlbnNvciIsImZlZWRzIiwiaW5wdXQiLCJydW4iLCJyZXN1bHRzIiwiZGF0YU91dCIsImRhdGEiLCJzbGljZSIsIk1hdGgiLCJtYXgiLCJyb3VuZCIsIlByb21pc2UiLCJyIiwic2V0VGltZW91dCIsIm9ubWVzc2FnZSIsImV2ZW50Iiwib3J0QmFja2dyb3VuZCIsInRoZW4iLCJyZXN1bHQiLCJfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX18iLCJfX3dlYnBhY2tfcmVxdWlyZV9fIiwibW9kdWxlSWQiLCJjYWNoZWRNb2R1bGUiLCJ1bmRlZmluZWQiLCJfX3dlYnBhY2tfbW9kdWxlc19fIiwiY2FsbCIsIm0iLCJ4IiwiX193ZWJwYWNrX2V4cG9ydHNfXyIsIk8iLCJjaHVua0lkcyIsImZuIiwicHJpb3JpdHkiLCJub3RGdWxmaWxsZWQiLCJJbmZpbml0eSIsImZ1bGZpbGxlZCIsImoiLCJPYmplY3QiLCJrZXlzIiwiZXZlcnkiLCJrZXkiLCJzcGxpY2UiLCJkIiwiZGVmaW5pdGlvbiIsIm8iLCJkZWZpbmVQcm9wZXJ0eSIsImVudW1lcmFibGUiLCJmIiwiZSIsImNodW5rSWQiLCJhbGwiLCJyZWR1Y2UiLCJwcm9taXNlcyIsInUiLCJnIiwiZ2xvYmFsVGhpcyIsInRoaXMiLCJGdW5jdGlvbiIsIndpbmRvdyIsIm9iaiIsInByb3AiLCJwcm90b3R5cGUiLCJoYXNPd25Qcm9wZXJ0eSIsIlN5bWJvbCIsInRvU3RyaW5nVGFnIiwidmFsdWUiLCJzY3JpcHRVcmwiLCJpbXBvcnRTY3JpcHRzIiwibG9jYXRpb24iLCJkb2N1bWVudCIsImN1cnJlbnRTY3JpcHQiLCJzcmMiLCJzY3JpcHRzIiwiZ2V0RWxlbWVudHNCeVRhZ05hbWUiLCJFcnJvciIsInJlcGxhY2UiLCJwIiwiaW5zdGFsbGVkQ2h1bmtzIiwiY2h1bmtMb2FkaW5nR2xvYmFsIiwicGFyZW50Q2h1bmtMb2FkaW5nRnVuY3Rpb24iLCJiaW5kIiwibW9yZU1vZHVsZXMiLCJydW50aW1lIiwicG9wIl0sInNvdXJjZVJvb3QiOiIifQ==