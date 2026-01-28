import sys
from train_module import run_training
from shap_module import run_shap
from design_module import run_inverse_design


def get_user_input_targets():
    """
    Auxiliary function: Obtain and verify the three target parameters input by the user
    """
    print("\n Please enter three target performance parameters, separated by spaces or commas.")
    print("Yield strength (MPa) Tensile strength (MPa) Elongation (%)")
    print("example: 650 680 10.5")

    while True:
        user_in = input("Please enter the target. >>> ").strip()

        user_in = user_in.replace('，', ',')


        if ',' in user_in:
            parts = user_in.split(',')
        else:
            parts = user_in.split()

        try:

            values = [float(x) for x in parts]

            if len(values) != 3:
                print(f"Error: Three numerical values need to be entered. and {len(values)} have been detected.Please try again.。")
                continue


            if any(v < 0 for v in values):
                print("Warning: Negative values have been detected, which may be actually unreasonable.")

            return values

        except ValueError:
            print("Error: The input contains non-numeric characters. Please ensure that only numerical values are entered.")


def main():
    while True:
        print("\n" + "=" * 40)
        print(" Aluminum alloy composition process deep learning and reverse design system")
        print("=" * 40)
        print("1. Train Model")
        print("2. SHAP  (SHAP Analysis)")
        print("3. Inverse Design - [Please enter the target manually.]")
        print("4. Run All - [Use the default target]")
        print("0. Exit")
        print("-" * 40)

        choice = input("Please enter the option number.: ").strip()

        if choice == '1':
            run_training()
        elif choice == '2':
            run_shap()
        elif choice == '3':

            target_vals = get_user_input_targets()

            run_inverse_design(target_values=target_vals)
        elif choice == '4':
            run_training()
            run_shap()

            run_inverse_design(target_values=None)
        elif choice == '0':
            print("exit")
            break
        else:
            print("Invalid input. Please select again.")


if __name__ == "__main__":
    main()