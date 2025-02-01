from django.shortcuts import render
import random
import string
import numpy as np

# Create your views here.

def home(request):
    return render(request, 'home.html', {})



def caesar_encrypt(request):

    if request.method == 'POST':

        plain_text = request.POST.get('inputText')
        plain = plain_text

        encrypted = []
        for char in plain:
            if char.isupper():
                new_char = chr((ord(char) - ord('A') + 3) % 26 + ord('A'))
                encrypted.append(new_char)
            elif char.islower():
                new_char = chr((ord(char) - ord('a') + 3) % 26 + ord('a'))
                encrypted.append(new_char)
            else:
                encrypted.append(char)
            
        string = ''.join(encrypted)
        cipher_text = string

        print('')
        print('----- Cipher Text: ', cipher_text, '-----')
        print('')

        return render(request, 'caesar.html', {'cipher_text': cipher_text})

    else:
        return render(request, 'caesar.html', {})



def decrypt_caesar(request):

    decrypted_text = []
    if request.method == 'POST':
        cipher_text = request.POST.get('encryptedCipher')
        for char in cipher_text:
            if char.isupper():
                new_char = chr((ord(char) - ord('A') - 3) % 26 + ord('A'))
                decrypted_text.append(new_char)

            elif char.islower():
                new_char = chr((ord(char) - ord('a') - 3) % 26 + ord('a'))
                decrypted_text.append(new_char)
            else:
                decrypted_text.append(char)

            decrypt_string = ''.join(decrypted_text)
            plain_text = decrypt_string

        print('')
        print('----- Plain Text: ', plain_text, '-----')
        print('')
        return render(request, 'caesar_decrypt.html', {'plain_text': plain_text})
        
    else:
        return render(request, 'caesar_decrypt.html', {})





#playfair cipher

def generate_playfair_square(key):

    key = key.upper().replace('J', 'I') 
    alphabet = string.ascii_uppercase.replace('J', '') 
    used_chars = set()
    square = []

    for char in key:
        if char not in used_chars and char in alphabet:
            square.append(char)
            used_chars.add(char)

    for char in alphabet:
        if char not in used_chars:
            square.append(char)

    return [square[i:i + 5] for i in range(0, 25, 5)]



def find_position(square, char):
 
    for row_idx, row in enumerate(square):
        if char in row:
            return row_idx, row.index(char)

    return None



def prepare_text(text):
    
    text = text.upper().replace('J', 'I') 
    prepared_text = []
    non_alpha_positions = {}  

    for idx, char in enumerate(text):
        if char.isalpha():
            if prepared_text and prepared_text[-1] == char:  
                prepared_text.append('X') 
            prepared_text.append(char)
        else:
            non_alpha_positions[idx] = char  

    if len(prepared_text) % 2 != 0:
        prepared_text.append('X')

    return ''.join(prepared_text), non_alpha_positions



def playfair_encrypt(request):

    key = "MONARCHY"
    if request.method == 'POST':
        plain_text = request.POST.get('inputTextPF')
        print("Plain Text:", plain_text)
        square = generate_playfair_square(key)
        prepared_text, non_alpha_positions = prepare_text(plain_text)

        encrypted_text = []

        for i in range(0, len(prepared_text), 2):
            a, b = prepared_text[i], prepared_text[i + 1]
            row_a, col_a = find_position(square, a)
            row_b, col_b = find_position(square, b)

            if row_a == row_b:  
                encrypted_text.append(square[row_a][(col_a + 1) % 5])
                encrypted_text.append(square[row_b][(col_b + 1) % 5])
            elif col_a == col_b: 
                encrypted_text.append(square[(row_a + 1) % 5][col_a])
                encrypted_text.append(square[(row_b + 1) % 5][col_b])
            else: 
                encrypted_text.append(square[row_a][col_b])
                encrypted_text.append(square[row_b][col_a])

        encrypted_with_spaces = list(encrypted_text)
        for pos, char in non_alpha_positions.items():
            encrypted_with_spaces.insert(pos, char)

        cipher_text = ''.join(encrypted_with_spaces)
        print('')
        print('----- Cipher Text: ', cipher_text, '-----')
        print('')
        print('----- Key: ', key, '-----')
        print('')
        return render(request, 'playfair.html', {'cipher_text': cipher_text, 'key': key})

    else:
        return render(request, 'playfair.html', {})



def playfair_decrypt(request):

    if request.method == 'POST':
        cipher_text = request.POST.get('cipherTextPF')
        key = request.POST.get('key')
        square = generate_playfair_square(key)
        print("Cipher Text:", cipher_text)

        non_alpha_positions = {idx: char for idx, char in enumerate(cipher_text) if not char.isalpha()}

        clean_cipher_text = ''.join([char for char in cipher_text if char.isalpha()])
        decrypted_text = []

        for i in range(0, len(clean_cipher_text), 2):
            a, b = clean_cipher_text[i], clean_cipher_text[i + 1]
            row_a, col_a = find_position(square, a)
            row_b, col_b = find_position(square, b)

            if row_a == row_b:  
                decrypted_text.append(square[row_a][(col_a - 1) % 5])
                decrypted_text.append(square[row_b][(col_b - 1) % 5])
            elif col_a == col_b: 
                decrypted_text.append(square[(row_a - 1) % 5][col_a])
                decrypted_text.append(square[(row_b - 1) % 5][col_b])
            else:  
                decrypted_text.append(square[row_a][col_b])
                decrypted_text.append(square[row_b][col_a])

        decrypted_with_spaces = list(decrypted_text)
        for pos, char in non_alpha_positions.items():
            decrypted_with_spaces.insert(pos, char)

        decrypted_string = ''.join(decrypted_with_spaces).rstrip('X') 
        print('')
        print('----- Plain Text: ', decrypted_string, '-----')
        print('')
        return render(request, 'playfair_decrypt.html', {'plain_text': decrypted_string})

    else:
        return render(request, 'playfair_decrypt.html', {})






#vigenere cipher

def vigenere_encrypt(request):
    key = "SNOW"
    if request.method == 'POST':
        plain_text = request.POST.get('inputTextVigenere')
        encrypted_text = []
        key_index = 0
        key = key.upper() 
        
        for char in plain_text:
            if char.isalpha():
                offset = ord('A') if char.isupper() else ord('a')
                new_char = chr((ord(char) - offset + ord(key[key_index]) - ord('A')) % 26 + offset)
                encrypted_text.append(new_char)
                key_index = (key_index + 1) % len(key)
            else:
                encrypted_text.append(char)
        
        cipher_text = ''.join(encrypted_text)
        print('')
        print('----- Cipher Text: ', cipher_text, '-----')
        print('')
        print('----- Key: ', key, '-----')
        print('')
        return render(request, 'vigenere.html', {'cipher_text': cipher_text, 'key': key})

    else:
        return render(request, 'vigenere.html', {})



def vigenere_decrypt(request):
    decrypted_text = []

    if request.method == 'POST':
        key = request.POST.get('key')
        cipher_text = request.POST.get('encryptedTextVigenere')
        key_index = 0
        key = key.upper()
    
        for char in cipher_text:
            if char.isalpha():
                offset = ord('A') if char.isupper() else ord('a')
                new_char = chr((ord(char) - offset - (ord(key[key_index]) - ord('A'))) % 26 + offset)
                decrypted_text.append(new_char)
                key_index = (key_index + 1) % len(key)
            else:
                decrypted_text.append(char)
        
        decrypt_string = ''.join(decrypted_text)
        plain_text = decrypt_string
        print('')
        print('----- Plain Text: ', plain_text, '-----')
        print('')
        return render(request, 'vigenere_decrypt.html', {'plain_text': plain_text})

    else:
        return render(request, 'vigenere_decrypt.html', {})






# OTP Cipher

def generate_otp_key(length):
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(length))


def encrypt_otp(request):

    encrypted_text = []
    if request.method == 'POST':

        plaintext = request.POST.get('inputTextOTP')
        key = generate_otp_key(len(plaintext))
        print('***** Random Key Generated: ', key, ' *****')

        for p, k in zip(plaintext, key):
            p_val = ord(p.upper()) - ord('A')
            k_val = ord(k.upper()) - ord('A')
            c_val = (p_val + k_val) % 26
            encrypted_text.append(chr(c_val + ord('A')))
        
        cipher_text = ''.join(encrypted_text)
        print('')
        print('----- Cipher Text: ', cipher_text, '-----')
        print('')
        print('----- Key: ', key, '-----')
        print('')
        return render(request, 'otp.html', {'cipher_text': cipher_text, 'key': key}) 

    else:
        return render(request, 'otp.html', {})



def decrypt_otp(request):

    plaintext = []
    if request.method == 'POST':
        ciphertext = request.POST.get('encryptedTextOTP')
        key = request.POST.get('key')

        for c, k in zip(ciphertext, key):
            c_val = ord(c.upper()) - ord('A')
            k_val = ord(k.upper()) - ord('A')
            p_val = (c_val - k_val + 26) % 26
            plaintext.append(chr(p_val + ord('A')))
        
        decrypt_string = ''.join(plaintext)
        plain_text = decrypt_string
        print('')
        print('----- Plain Text: ', plain_text, '-----')
        print('')
        return render(request, 'otp_decrypt.html', {'plain_text': plain_text})

    else:
        return render(request, 'otp_decrypt.html', {})




# row-column cipher
def row_column_encrypt(request):
   
    if request.method == 'POST':
        plain_text = request.POST.get('inputTextRC')
        key = 'ZEBRAS'
        key = key.upper()
        key_length = len(key)
        sorted_key_indices = sorted(range(key_length), key=lambda k: key[k])

        plain_text = plain_text.replace(" ", "")
        rows = [plain_text[i:i + key_length] for i in range(0, len(plain_text), key_length)]

        last_row = rows[-1]
        if len(last_row) < key_length:
            rows[-1] = last_row + " " * (key_length - len(last_row))

        cipher_text = ""
        for col in sorted_key_indices:
            cipher_text += "".join(row[col] for row in rows)

        print('')
        print('----- Cipher Text: ', cipher_text, '-----')
        print('')
        print('----- Key: ', key, '-----')
        print('')
        return render(request, 'row_column.html', {'cipher_text': cipher_text, 'key': key})

    else:
        return render(request, 'row_column.html', {})



def row_column_decrypt(request):

    if request.method == 'POST':
        key = request.POST.get('key') or "ZEBRAS" 
        cipher_text = request.POST.get('encryptedTextRC').replace(" ", "") 
        key = key.upper()
        key_length = len(key)
        sorted_key_indices = sorted(range(key_length), key=lambda k: key[k])

        num_rows = len(cipher_text) // key_length

        grid = [[""] * key_length for _ in range(num_rows)]

        col_length = num_rows
        start = 0

        for col_index in sorted_key_indices:
            for row in range(col_length):
                grid[row][col_index] = cipher_text[start]
                start += 1

        plain_text = "".join("".join(row) for row in grid).rstrip()
        print('')
        print('----- Plain Text: ', plain_text, '-----')
        print('')

        return render(request, 'rc_decrypt.html', {'plain_text': plain_text})

    else:
        return render(request, 'rc_decrypt.html', {})




# Hill Cipher

def hill_encrypt(request):
    
    key = np.array([[3, 3], [2, 5]]) 
    if request.method == 'POST':
        plaintext = request.POST.get('inputTextHill')
        plaintext = plaintext.replace(" ", "").upper()
        n = key.shape[0] 

        while len(plaintext) % n != 0:
            plaintext += 'X'

        print("Plaintext:", plaintext)

        plaintext_vector = [ord(char) - ord('A') for char in plaintext]

        print("Plaintext Vector:", plaintext_vector)
        plaintext_blocks = np.array(plaintext_vector).reshape(-1, n)

        print("Plaintext Blocks:")
        print(plaintext_blocks)

        encrypted_blocks = np.dot(plaintext_blocks, key) % 26

        print("Encrypted Blocks:")
        print(encrypted_blocks)

        cipher_text = ''
        for block in encrypted_blocks:
            for num in block:
                cipher_text += chr(num + ord('A'))

        print('')
        print('----- Cipher Text: ', cipher_text, '-----')
        print('')
        print('----- Key: ', key, '-----')
        print('')
        return render(request, 'hill.html', {'cipher_text': cipher_text, 'key': key})

    else:
        return render(request, 'hill.html', {})



def hill_decrypt(request):
    
    if request.method == 'POST':
        print("Hill Cipher Decrypt View Triggered")

        key_input = request.POST.get('key')
        cipher_text = request.POST.get('encryptedTextHill')  

        print("Cipher Text:", cipher_text)
        print("Key Input:", key_input)

        if not key_input or not cipher_text:
            print("Missing key or ciphertext")
            return render(request, 'hill_decrypt.html', {'error': 'Please provide both key and ciphertext.'})

        cipher_text = cipher_text.replace(" ", "").upper()

        try:
            key_input = key_input.replace('[', '').replace(']', '').replace(' ', ',')
            key_list = list(map(int, key_input.split(',')))
            n = int(len(key_list) ** 0.5)  
            key = np.array(key_list).reshape(n, n)  
        except Exception as e:
            print("Key Parsing Error:", e)
            return render(request, 'hill_decrypt.html', {'error': 'Invalid key format. Please provide a valid matrix.'})

        print("Parsed Key Matrix:\n", key)

        if key.shape[0] != key.shape[1]:
            print("Key is not a square matrix")
            return render(request, 'hill_decrypt.html', {'error': 'Key must be a square matrix.'})

        try:
            determinant = int(np.round(np.linalg.det(key))) 
            determinant_inv = pow(determinant, -1, 26) 
            key_inverse = (
                determinant_inv * np.round(determinant * np.linalg.inv(key)).astype(int) % 26
            )  
        except Exception as e:
            print("Matrix Inversion Error:", e)
            return render(request, 'hill_decrypt.html', {'error': 'Key matrix is not invertible modulo 26.'})

        print("Key Inverse Matrix:\n", key_inverse)
        cipher_vector = [ord(char) - ord('A') for char in cipher_text]
        print("Cipher Vector:", cipher_vector)

        try:
            cipher_blocks = np.array(cipher_vector).reshape(-1, key.shape[0])
            print("Cipher Blocks:\n", cipher_blocks)
        except Exception as e:
            print("Cipher Blocks Reshape Error:", e)
            return render(request, 'hill_decrypt.html', {'error': 'Cipher text length is invalid for the key matrix.'})

        decrypted_blocks = (np.dot(cipher_blocks, key_inverse) % 26)
        print("Decrypted Blocks:\n", decrypted_blocks)

        plain_text = ''.join(chr(int(num) + ord('A')) for num in decrypted_blocks.flatten())
        print("Decrypted Plain Text:", plain_text)

        return render(request, 'hill_decrypt.html', {'plain_text': plain_text, 'key': key.tolist()})

    else:
        return render(request, 'hill_decrypt.html', {})




# def hill_decrypt(request):
#     """
#     Decrypts the ciphertext using Hill Cipher with the provided key matrix.
#     """
#     if request.method == 'POST':
#         key_input = request.POST.get('key')  # Get the key matrix as a string from the user
#         cipher_text = request.POST.get('encryptedTextHill')  # Get ciphertext input

#         # Normalize inputs
#         cipher_text = cipher_text.replace(" ", "").upper()

#         # Convert the key input string into a numpy array
#         try:
#             # Example input format for key: "3,3,2,5"
#             key_list = list(map(int, key_input.split(',')))  # Convert string to a list of integers
#             n = int(len(key_list) ** 0.5)  # Determine the dimension of the matrix (e.g., 2x2 for 4 elements)
#             key = np.array(key_list).reshape(n, n)  # Reshape into an n x n numpy array
#         except Exception as e:
#             return render(request, 'hill_decrypt.html', {'error': 'Invalid key format. Please provide a valid matrix.'})

#         # Ensure the key matrix is square
#         if key.shape[0] != key.shape[1]:
#             return render(request, 'hill_decrypt.html', {'error': 'Key must be a square matrix.'})

#         # Calculate the inverse of the key matrix modulo 26
#         try:
#             determinant = int(np.round(np.linalg.det(key)))  # Determinant of the key matrix
#             determinant_inv = pow(determinant, -1, 26)  # Modular multiplicative inverse of determinant mod 26
#             key_inverse = (
#                 determinant_inv * np.round(determinant * np.linalg.inv(key)).astype(int) % 26
#             )  # Inverse matrix mod 26
#         except Exception as e:
#             return render(request, 'hill_decrypt.html', {'error': 'Key matrix is not invertible modulo 26.'})

#         # Convert ciphertext to numerical values (A=0, B=1, ..., Z=25)
#         cipher_vector = [ord(char) - ord('A') for char in cipher_text]

#         # Reshape ciphertext vector into blocks of size n
#         cipher_blocks = np.array(cipher_vector).reshape(-1, key.shape[0])

#         # Decrypt: Multiply inverse key matrix with each block and apply modulo 26
#         decrypted_blocks = (np.dot(cipher_blocks, key_inverse) % 26)

#         # Convert decrypted numbers back to letters
#         plain_text = ''.join(chr(int(num) + ord('A')) for num in decrypted_blocks.flatten())
#         print(plain_text)
#         print(cipher_text)
#         print(key)

#         return render(request, 'hill_decrypt.html', {'plain_text': plain_text, 'key': key.tolist()})

#     else:
#         return render(request, 'hill_decrypt.html', {})


# def hill_decrypt(request):
#     """
#     Decrypts the ciphertext using Hill Cipher with the provided key matrix.
#     """
#     key = np.array([[3, 3], [2, 5]])  # Ensure the matrix is invertible modulo 26
#     if request.method == 'POST':
#         ciphertext = request.POST.get('encryptedTextHill')
#         key = request.POST.get('key')
#         ciphertext = ciphertext.replace(" ", "").upper()
#         n = key.shape[0]  # Dimension of the key matrix

#         # Calculate the modular inverse of the key matrix
#         det_key = int(round(np.linalg.det(key))) % 26
#         inv_det_key = pow(det_key, -1, 26)
#         adj_key = np.array([[key[1, 1], -key[0, 1]], [-key[1, 0], key[0, 0]]])
#         inv_key = (inv_det_key * adj_key) % 26

#         # Convert ciphertext to numerical form (A=0, B=1, ..., Z=25)
#         ciphertext_vector = [ord(char) - ord('A') for char in ciphertext]

#         # Split into blocks of size n
#         ciphertext_blocks = np.array(ciphertext_vector).reshape(-1, n)

#         # Decrypt each block: Plaintext = (Cipher * InverseKey) % 26
#         decrypted_blocks = (np.dot(ciphertext_blocks, inv_key) % 26)

#         # Convert numerical form back to characters
#         plain_text = ''.join(chr(int(num) + ord('A')) for num in decrypted_blocks.flatten())

#         print('')
#         print('----- Plain Text: ', plain_text, '-----')
#         print('')
#         print('----- Key: ', key, '-----')
#         print('')
#         return render(request, 'hill_decrypt.html', {'plain_text': plain_text, 'key': key})

#     else:
#         return render(request, 'hill_decrypt.html', {})




def encrypt(message, key):
    message = message.upper()
    key = key.upper()

    key_matrix = np.array([list(map(lambda x: ord(x) % 65, key[i:i+3])) for i in range(0, len(key), 3)])
    message += 'X' * (3 - len(message) % 3) if len(message) % 3 != 0 else ''

    cipher_text = ''
    for i in range(0, len(message), 3):
        message_vector = np.array(list(map(lambda x: ord(x) % 65, message[i:i+3])))
        cipher_vector = np.dot(message_vector, key_matrix.T) % 26
        cipher_text += ''.join(chr(c + 65) for c in cipher_vector)

    print("Ciphertext:", cipher_text)
    return cipher_text

def modinv(a, m):
    # Extended Euclidean Algorithm
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1

def decrypt(cipher_text, key):
    cipher_text = cipher_text.upper()
    key = key.upper()

    key_matrix = np.array([list(map(lambda x: ord(x) % 65, key[i:i+3])) for i in range(0, len(key), 3)])
    key_matrix_inv = np.linalg.inv(key_matrix) % 26

    plain_text = ''
    for i in range(0, len(cipher_text), 3):
        cipher_vector = np.array(list(map(lambda x: ord(x) % 65, cipher_text[i:i+3])))
        plain_vector = np.dot(cipher_vector, key_matrix_inv) % 26
        plain_text += ''.join(chr(int(c) % 26 + 65) for c in plain_vector)

    print("Decrypted Text:", plain_text)
    return plain_text

# Driver Code
def main():
    message = "GFG"
    key = "GYBNQKURP"

    cipher_text = encrypt(message, key)

    # Use the same key for decryption
    decrypted_text = decrypt(cipher_text, key)

if __name__ == "__main__":
    main()





# def hill_encrypt(request):
#     """
#     Encrypts the plaintext using Hill Cipher with the provided key matrix.
#     """
#     key = np.array([[3, 3], [2, 5]])  # Ensure the matrix is invertible modulo 26
#     if request.method == 'POST':
#         plaintext = request.POST.get('inputTextHill')
#         plaintext = plaintext.replace(" ", "").upper()
#         n = key.shape[0]  # Dimension of the key matrix

#         # Padding the plaintext to fit the key matrix dimensions
#         while len(plaintext) % n != 0:
#             plaintext += 'X'

#         # Convert plaintext to numerical form (A=0, B=1, ..., Z=25)
#         plaintext_vector = [ord(char) - ord('A') for char in plaintext]

#         # Split into blocks of size n
#         plaintext_blocks = np.array(plaintext_vector).reshape(-1, n)

#         # Encrypt each block: Cipher = (Key * Plaintext) % 26
#         encrypted_blocks = (np.dot(plaintext_blocks, key) % 26)

#         # Convert numerical form back to characters
#         cipher_text = ''.join(chr(int(num) + ord('A')) for num in encrypted_blocks.flatten())

#         # return cipher_text
#         print('')
#         print('----- Cipher Text: ', cipher_text, '-----')
#         print('')
#         print('----- Key: ', key, '-----')
#         print('')
#         return render(request, 'hill.html', {'cipher_text': cipher_text, 'key': key})

#     else:
#         return render(request, 'hill.html', {})


# def hill_decrypt(ciphertext, key):
#     """
#     Decrypts the ciphertext using Hill Cipher with the provided key matrix.
#     """
#     ciphertext = ciphertext.replace(" ", "").upper()
#     n = key.shape[0]  # Dimension of the key matrix

#     # Convert ciphertext to numerical form
#     ciphertext_vector = [ord(char) - ord('A') for char in ciphertext]
#     ciphertext_blocks = np.array(ciphertext_vector).reshape(-1, n)

#     # Calculate the inverse key matrix modulo 26
#     det = int(round(np.linalg.det(key)))  # Determinant of the key matrix
#     det_inv = pow(det, -1, 26)  # Modular inverse of the determinant mod 26
#     key_inv = (det_inv * np.round(det * np.linalg.inv(key)).astype(int) % 26) % 26

#     # Decrypt each block: Plaintext = (InverseKey * Cipher) % 26
#     decrypted_blocks = (np.dot(ciphertext_blocks, key_inv) % 26)

#     # Convert numerical form back to characters
#     plain_text = ''.join(chr(int(num) + ord('A')) for num in decrypted_blocks.flatten())

#     return plain_text.rstrip('X')  # Remove padding


# Example Usage
# if __name__ == "__main__":
#     # Key matrix (2x2 example)
#     key = np.array([[3, 3], [2, 5]])  # Ensure the matrix is invertible modulo 26

#     # Encrypt
#     plaintext = "HELLO"
#     cipher_text = hill_encrypt(plaintext, key)
#     print(f"Encrypted Text: {cipher_text}")

#     # Decrypt
#     decrypted_text = hill_decrypt(cipher_text, key)
#     print(f"Decrypted Text: {decrypted_text}")




# Rail Fence

def rail_fence_encrypt(request):

    #Encrypts the plaintext using Rail Fence Cipher with a depth of 2.

    if request.method == 'POST':
        plain_text = request.POST.get('inputTextRF')

        rail_1 = []  
        rail_2 = []  

        for i, char in enumerate(plain_text):
            if i % 2 == 0:
                rail_1.append(char)
            else: 
                rail_2.append(char)

        cipher_text = ''.join(rail_1) + ''.join(rail_2)
        print('')
        print('----- Cipher Text: ', cipher_text, '-----')
        print('')
        return render(request, 'rail_fence.html', {'cipher_text': cipher_text})
    
    else:
        return render(request, 'rail_fence.html', {})



def rail_fence_decrypt(request):

    #Decrypts the ciphertext using Rail Fence Cipher with a depth of 2.

    if request.method == 'POST':
        cipher_text = request.POST.get('encryptedCipherRF')

        mid = (len(cipher_text) + 1) // 2 
        rail_1 = cipher_text[:mid]  
        rail_2 = cipher_text[mid:] 

        plain_text = []
        for r1, r2 in zip(rail_1, rail_2):
            plain_text.append(r1)  
            plain_text.append(r2)  

        if len(rail_1) > len(rail_2):
            plain_text.append(rail_1[-1])

        plain_text = ''.join(plain_text)
        print('')
        print('----- Plain Text: ', plain_text, '-----')
        print('')
        return render(request, 'rf_decrypt.html', {'plain_text': plain_text})

    else:
        return render(request, 'rf_decrypt.html', {})





